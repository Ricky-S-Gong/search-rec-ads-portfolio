"""Verify the FINAL selected architecture consumes prepared inputs; no update."""
import copy

import torch

from .model import loss_fn
from .sampling import batch_probability


def check(model,batch,spec,temperature):
    probability=batch_probability(model,batch['item_id'])
    previous=model.training
    model.eval();model.zero_grad(set_to_none=True)
    # cuDNN only permits LSTM backward in training mode.  The LSTM here has
    # no internal dropout, while the surrounding model remains in eval mode
    # so this zero-update acceptance check stays deterministic.
    if model.user.history_mode == 'lstm':
        model.user.lstm.train()
    u,v=model(batch)
    loss=loss_fn(u,v,batch['item_id'],temperature,probability)
    if not torch.isfinite(loss):raise ValueError('Nonfinite preflight loss')
    loss.backward()
    report={'parameter_updates':0,'item_features':{},'user_features':{},'item_mlp_input':model.item.input_dim,'user_mlp_input':model.user.input_dim,'user_shape':list(u.shape),'item_shape':list(v.shape),'history_encoder':model.user.history_mode}
    rows=model.item.rows(batch['item_id'])
    cat=model.item.categories[rows];num=model.item.numbers[rows]
    tags=model.item.tag_indices[rows] if model.item.enriched else None
    extra=model.item.extra_numbers[rows] if model.item.enriched else None
    with torch.no_grad():
        base=model.item.from_features(cat,num,tags,extra)
        for i,name in enumerate(spec['categorical_names']):
            changed=cat.clone();changed[:,i]=0
            changed_tags=torch.zeros_like(tags) if tags is not None and name=='tag' else tags
            delta=float((model.item.from_features(changed,num,changed_tags,extra)-base).abs().max())
            grad=model.item.embeddings[i].weight.grad
            norm=float(grad.norm()) if grad is not None else 0
            report['item_features'][name]={'output_change':delta,'gradient_norm':norm}
            if delta<=0 or norm<=0 or not torch.isfinite(torch.tensor(norm)):
                raise ValueError('Item feature failed participation check: '+name)
        for i,name in enumerate(spec['numeric_names']):
            changed=num.clone();changed[:,i]+=1
            delta=float((model.item.from_features(cat,changed,tags,extra)-base).abs().max())
            report['item_features'][name]={'output_change':delta}
            if delta<=0:raise ValueError('Numeric feature is unused: '+name)
        if model.item.enriched:
            for i,name in enumerate(model.feature_spec['item_numeric_names']):
                changed=extra.clone();changed[:,i]+=1
                delta=float((model.item.from_features(cat,num,tags,changed)-base).abs().max())
                report['item_features'][name]={'output_change':delta}
                if delta<=0:raise ValueError('Extra item feature unused: '+name)
            for i,name in enumerate(model.feature_spec['user_numeric_names']):
                altered={k:t.clone() for k,t in batch.items()};altered['user_numeric'][:,i]+=1
                delta=float((model.users(altered)-u).abs().max())
                report['user_features'][name]={'output_change':delta}
                if delta<=0:raise ValueError('Behavior summary unused: '+name)
            for i,name in enumerate(model.feature_spec['history_numeric_names']):
                altered={k:t.clone() for k,t in batch.items()};altered['history_feedback'][:,:,i]+=batch['history_mask']
                delta=float((model.users(altered)-u).abs().max())
                report['user_features'][name]={'output_change':delta}
                if delta<=0:raise ValueError('History feedback unused: '+name)
        for name in ['user_id','tab','time_ms']:
            altered={k:t.clone() for k,t in batch.items()}
            if name=='user_id':
                pos=torch.searchsorted(model.user.raw_ids,altered[name]);altered[name]=model.user.raw_ids[(pos+1)%len(model.user.raw_ids)]
            elif name=='tab':altered[name]=(altered[name]+1)%15
            else:altered[name]+=3600000
            delta=float((model.users(altered)-u).abs().max())
            report['user_features'][name]={'output_change':delta}
            if delta<=0:raise ValueError('User feature is unused: '+name)
        if model.user.history_mode!='none':
            altered={k:t.clone() for k,t in batch.items()}
            altered['history_mask'].zero_()
            delta=float((model.users(altered)-u).abs().max())
            if delta<=0:raise ValueError('History is unused')
            report['user_features']['history_mask']={'output_change':delta}
            altered={k:t.clone() for k,t in batch.items()}
            altered['history_item_ids']=torch.roll(altered['history_item_ids'],1,0)
            altered['history_mask']=torch.roll(altered['history_mask'],1,0)
            delta=float((model.users(altered)-u).abs().max())
            if delta<=0:raise ValueError('History item IDs are unused')
            report['user_features']['history_item_ids']={'output_change':delta}
    for name,embedding in [('user_id',model.user.user_embedding),('tab',model.user.tab_embedding)]:
        norm=float(embedding.weight.grad.norm()) if embedding.weight.grad is not None else 0
        report['user_features'][name]['gradient_norm']=norm
        if norm<=0 or not torch.isfinite(torch.tensor(norm)):raise ValueError('No finite user/context embedding gradient')
    # Verify continuous item inputs also have gradients through this model.
    numeric=num.detach().clone().requires_grad_(True)
    numeric_loss=loss_fn(u.detach(),model.item.from_features(cat,numeric,tags,extra),batch['item_id'],temperature,probability)
    grads=torch.autograd.grad(numeric_loss,numeric)[0]
    for i,name in enumerate(spec['numeric_names']):
        norm=float(grads[:,i].norm())
        report['item_features'][name]['gradient_norm']=norm
        if norm<=0 or not torch.isfinite(torch.tensor(norm)):raise ValueError('No finite numeric input gradient: '+name)
    if model.item.enriched:
        extra_input=extra.detach().clone().requires_grad_(True)
        loss=loss_fn(u.detach(),model.item.from_features(cat,num,tags,extra_input),batch['item_id'],temperature,probability)
        grads=torch.autograd.grad(loss,extra_input)[0]
        for i,name in enumerate(model.feature_spec['item_numeric_names']):
            norm=float(grads[:,i].norm());report['item_features'][name]['gradient_norm']=norm
            if norm<=0 or not torch.isfinite(torch.tensor(norm)):raise ValueError('Extra item gradient missing: '+name)
        altered={k:t.clone() for k,t in batch.items()}
        altered['user_numeric'].requires_grad_(True);altered['history_feedback'].requires_grad_(True)
        loss=loss_fn(model.users(altered),v.detach(),batch['item_id'],temperature,probability)
        grads=torch.autograd.grad(loss,[altered['user_numeric'],altered['history_feedback']])
        for names,g in [(model.feature_spec['user_numeric_names'],grads[0]),(model.feature_spec['history_numeric_names'],grads[1])]:
            for i,name in enumerate(names):
                norm=float(g[...,i].norm());report['user_features'][name]['gradient_norm']=norm
                if norm<=0 or not torch.isfinite(torch.tensor(norm)):raise ValueError('Behavior feature gradient missing: '+name)
        report['feature_version']=model.feature_spec['version']
        preference=model.preferences(batch).detach().requires_grad_(True)
        history=model.item(batch['history_item_ids']).detach()
        loss=loss_fn(model.user(batch,history,preference),v.detach(),batch['item_id'],temperature,probability)
        grad=torch.autograd.grad(loss,preference)[0]
        for name,a,b in [('author_preference',0,8),('tag_preference',8,16),('duration_preference',16,17)]:
            norm=float(grad[:,a:b].norm())
            with torch.no_grad():
                changed=preference.clone();changed[:,a:b]+=1
                delta=float((model.user(batch,history,changed)-u).abs().max())
            report['user_features'][name]={'output_change':delta,'gradient_norm':norm}
            if norm<=0 or delta<=0 or not torch.isfinite(torch.tensor(norm)):raise ValueError('Preference branch unused: '+name)
    report['passed']=True
    model.zero_grad(set_to_none=True);model.train(previous)
    return report

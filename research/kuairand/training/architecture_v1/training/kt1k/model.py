"""Content-aware item tower and mean/attention/LSTM user history encoders."""
import math

import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .data import load_item_data


class SafeBN(nn.BatchNorm1d):
    def forward(self,x):
        if self.training and len(x)==1:
            return F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,False,self.momentum,self.eps)
        return super().forward(x)


class Residual(nn.Module):
    def __init__(self,width,bn,drop):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(width,width),SafeBN(width) if bn else nn.Identity(),nn.ReLU(),nn.Dropout(drop),nn.Linear(width,width))
    def forward(self,x):
        return F.relu(x+self.net(x))


def mlp(input_dim,hidden,output,settings):
    layers=[]
    width=input_dim
    for n in hidden:
        layers.extend([nn.Linear(width,n),SafeBN(n) if settings['batch_norm'] else nn.Identity(),nn.ReLU(),nn.Dropout(settings['dropout'])])
        if settings['residual_blocks']:
            layers.extend(Residual(n,settings['batch_norm'],settings['dropout']) for _ in range(settings['residual_blocks']))
        width=n
    layers.append(nn.Linear(width,output))
    return nn.Sequential(*layers)


class ItemTower(nn.Module):
    def __init__(self,root,spec,settings,feature_spec=None):
        super().__init__()
        raw,cat,num=load_item_data(root,spec)
        # Immutable features are rebuilt from hash-bound data; do not duplicate
        # hundreds of MB of static buffers in every model checkpoint.
        self.register_buffer('raw_ids',torch.tensor(raw),persistent=False)
        self.register_buffer('categories',torch.tensor(np.vstack([np.zeros((1,cat.shape[1]),dtype=np.int64),cat])),persistent=False)
        self.register_buffer('numbers',torch.tensor(np.vstack([np.zeros((1,num.shape[1]),dtype=np.float32),num])),persistent=False)
        self.embeddings=nn.ModuleList(nn.Embedding(card,settings['category_dim'],padding_idx=0) for card in spec['cardinalities'])
        self.input_dim=len(spec['categorical_names'])*settings['category_dim']+len(spec['numeric_names'])
        self.enriched=feature_spec is not None
        if self.enriched:
            import pyarrow.parquet as pq
            extra=pq.read_table(Path(root)/'features-v2'/'item_extra.parquet')
            if not np.array_equal(extra['item_id'].to_numpy(),raw):raise ValueError('Item extension identity mismatch')
            tag=extra['tag_indices'].to_pylist()
            tag_array=np.zeros((len(raw)+1,feature_spec['max_tags']),dtype=np.int32)
            for i,values in enumerate(tag):tag_array[i+1,:len(values)]=values
            nums=np.column_stack([extra['num_'+n].to_numpy() for n in feature_spec['item_numeric_names']]).astype(np.float32)
            if not np.isfinite(nums).all() or tag_array.max()>=feature_spec['tag_cardinality']:
                raise ValueError('Invalid extra item values')
            self.register_buffer('tag_indices',torch.tensor(tag_array),persistent=False)
            self.register_buffer('extra_numbers',torch.tensor(np.vstack([np.zeros((1,4),dtype=np.float32),nums])),persistent=False)
            self.embeddings[-1]=nn.Embedding(feature_spec['tag_cardinality'],settings['category_dim'],padding_idx=0)
            self.input_dim+=4
        self.output_dim=settings['output_dim']
        self.net=mlp(self.input_dim,settings['item_hidden'],settings['output_dim'],settings)

    def rows(self,ids):
        pos=torch.searchsorted(self.raw_ids,ids.contiguous())
        safe=pos.clamp(max=len(self.raw_ids)-1)
        valid=(ids>=0)&(pos<len(self.raw_ids))&(self.raw_ids[safe]==ids)
        if torch.any((ids>=0)&~valid):
            raise ValueError('Unknown raw item ID, not an unseen category')
        return torch.where(valid,pos+1,0)

    def tag_vectors(self,tags):
        mask=tags>0
        return (self.embeddings[-1](tags.long())*mask.unsqueeze(-1)).sum(-2)/mask.sum(-1).clamp(min=1).unsqueeze(-1)

    def from_features(self,cat,num,tags=None,extra=None):
        shape=cat.shape[:-1]
        cat=cat.reshape(-1,cat.shape[-1]);num=num.reshape(-1,num.shape[-1])
        if self.enriched:
            tags=tags.reshape(len(cat),-1);extra=extra.reshape(len(cat),-1)
            parts=[e(cat[:,i]) for i,e in enumerate(self.embeddings[:-1])]+[self.tag_vectors(tags),num.float(),extra.float()]
        else:parts=[e(cat[:,i]) for i,e in enumerate(self.embeddings)]+[num.float()]
        x=torch.cat(parts,dim=-1)
        return F.normalize(self.net(x).float(),dim=-1).reshape(*shape,-1)

    def forward(self,ids):
        rows=self.rows(ids)
        flat=rows.reshape(-1);valid=flat>0
        result=torch.zeros((len(flat),self.output_dim),device=ids.device)
        if valid.any():
            chosen=flat[valid]
            result[valid]=self.from_features(self.categories[chosen],self.numbers[chosen],self.tag_indices[chosen] if self.enriched else None,self.extra_numbers[chosen] if self.enriched else None)
        return result.reshape(*ids.shape,self.output_dim)


class UserTower(nn.Module):
    def __init__(self,user_ids,settings,feature_spec=None):
        super().__init__()
        self.register_buffer('raw_ids',torch.tensor(user_ids,dtype=torch.long),persistent=False)
        self.user_embedding=nn.Embedding(len(user_ids)+1,settings['user_dim'])
        self.tab_embedding=nn.Embedding(16,settings['tab_dim'],padding_idx=0)
        self.history_mode=settings['history_encoder']
        out=settings['output_dim']
        self.context_dim=settings['user_dim']+settings['tab_dim']+4
        self.enriched=feature_spec is not None
        if self.enriched:
            self.context_dim+=len(feature_spec['user_numeric_names'])
            self.feedback_projection=nn.Linear(5,8)
            self.feedback_gate=nn.Linear(5,1)
            self.history_projection=nn.Linear(out+8,out)
        self.input_dim=self.context_dim+(out if self.history_mode!='none' else 0)
        if self.enriched:self.input_dim+=settings['category_dim']*2+1
        if self.history_mode=='attention':
            self.query=nn.Linear(self.context_dim,out)
            self.position=nn.Embedding(settings['history_length'],out)
            self.attention=nn.MultiheadAttention(out,settings['attention_heads'],dropout=settings['dropout'],batch_first=True)
        elif self.history_mode=='lstm':
            self.lstm=nn.LSTM(out,out,batch_first=True)
        elif self.history_mode not in ['mean','none']:
            raise ValueError('Unknown history encoder')
        self.net=mlp(self.input_dim,settings['user_hidden'],out,settings)

    def context(self,batch):
        ids=batch['user_id'];pos=torch.searchsorted(self.raw_ids,ids)
        safe=pos.clamp(max=len(self.raw_ids)-1)
        indices=torch.where((pos<len(self.raw_ids))&(self.raw_ids[safe]==ids),pos+1,0)
        tabs=batch['tab'];tab_indices=torch.where((tabs>=0)&(tabs<15),tabs+1,0)
        sec=batch['time_ms'].double()/1000
        day=torch.remainder(sec,86400)*(2*math.pi/86400)
        week=torch.remainder(sec+3*86400,7*86400)*(2*math.pi/(7*86400))
        parts=[self.user_embedding(indices),self.tab_embedding(tab_indices),torch.stack([day.sin(),day.cos(),week.sin(),week.cos()],1).float()]
        if self.enriched:parts.append(batch['user_numeric'])
        return torch.cat(parts,1)

    def forward(self,batch,history=None,preference=None):
        context=self.context(batch)
        if self.enriched and self.history_mode!='none':
            feedback=F.relu(self.feedback_projection(batch['history_feedback']))
            history=self.history_projection(torch.cat([history,feedback],-1))*torch.sigmoid(self.feedback_gate(batch['history_feedback']))*2
        if self.history_mode=='none':
            return F.normalize(self.net(torch.cat([context,preference],1) if self.enriched else context).float(),dim=-1)
        mask=batch['history_mask'];lengths=mask.sum(1)
        pooled=torch.zeros((len(context),history.shape[-1]),device=context.device)
        if self.history_mode=='mean':
            pooled=(history*mask.unsqueeze(-1)).sum(1)/lengths.clamp(min=1).unsqueeze(-1)
        else:
            valid=lengths>0
            if valid.any():
                h=history[valid];m=mask[valid]
                if self.history_mode=='attention':
                    position=self.position(torch.arange(h.shape[1],device=h.device))[None,:,:]
                    value,_=self.attention(self.query(context[valid]).unsqueeze(1),h+position,h,key_padding_mask=~m,need_weights=False)
                    pooled[valid]=value.squeeze(1).to(pooled.dtype)
                else:
                    # pack_sequence removes LEFT padding and retains order.
                    seq=nn.utils.rnn.pack_sequence([a[b] for a,b in zip(h,m)],enforce_sorted=False)
                    _,(hidden,_)=self.lstm(seq)
                    pooled[valid]=hidden[-1].to(pooled.dtype)
        parts=[context,pooled]
        if self.enriched:parts.append(preference)
        return F.normalize(self.net(torch.cat(parts,1)).float(),dim=-1)


class TwoTower(nn.Module):
    def __init__(self,root,spec,user_ids,settings,feature_spec=None):
        super().__init__()
        self.item=ItemTower(root,spec,settings,feature_spec)
        self.user=UserTower(user_ids,settings,feature_spec)
    def preferences(self,batch):
        rows=self.item.rows(batch['history_item_ids']);mask=batch['history_mask'].unsqueeze(-1)
        author=self.item.embeddings[1](self.item.categories[rows,1])
        tags=self.item.tag_vectors(self.item.tag_indices[rows])
        duration=self.item.numbers[rows,0].unsqueeze(-1)
        return (torch.cat([author,tags,duration],-1)*mask).sum(1)/mask.sum(1).clamp(min=1)
    def users(self,batch):
        history=self.item(batch['history_item_ids']) if self.user.history_mode!='none' else None
        preference=self.preferences(batch) if self.user.enriched else None
        return self.user(batch,history,preference)
    def forward(self,batch):
        return self.users(batch),self.item(batch['item_id'])


def loss_fn(users,items,ids,temperature,candidate_probability=None,logq_alpha=1.0):
    logits=users.float()@items.float().T/temperature
    if candidate_probability is not None:
        if isinstance(logq_alpha,bool) or not isinstance(logq_alpha,(int,float)) or not 0 <= logq_alpha <= 1:
            raise ValueError('logq_alpha must be a number in [0, 1]')
        q=candidate_probability.float()
        if q.shape!=(len(ids),) or not torch.isfinite(q).all() or (q<=0).any() or (q>1).any():
            raise ValueError('Candidate sampling probabilities must be a finite positive vector <= 1')
        # Column correction AFTER temperature scaling; evaluation uses raw dot products.
        logits=logits-float(logq_alpha)*q.log().unsqueeze(0)
    repeated=ids[:,None]==ids[None,:]
    repeated.fill_diagonal_(False)
    return F.cross_entropy(logits.masked_fill(repeated,float('-inf')),torch.arange(len(ids),device=ids.device))

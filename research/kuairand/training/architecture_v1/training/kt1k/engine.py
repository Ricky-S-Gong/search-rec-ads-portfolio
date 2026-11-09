"""Manual training, every-epoch exhaustive validation, best/latest checkpoints."""
import hashlib
import json
import os
import platform
import random
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .acceptance import check
from .data import Queries,sha,verify,extension
from .evaluation import evaluate
from .model import TwoTower,loss_fn
from .sampling import configure_sampling,batch_probability
from .recording import validation_loss,sync_epoch_artifacts


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def json_write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def atomic_save(path,state):
    temp=path.with_suffix('.pending');torch.save(state,temp);temp.replace(path)


def code_hash():
    root=Path(__file__).parent
    return hashlib.sha256(json.dumps([(p.name,sha(p)) for p in sorted(root.glob('*.py'))]).encode()).hexdigest()


def load(root,config):
    import pyarrow.parquet as pq
    root=Path(root);manifest=verify(root);spec=json.loads((root/'encoding_spec.json').read_text())
    model_config=config['model']
    patience=config['training'].get('early_stopping_patience',0)
    if type(patience) is not int or patience<0:
        raise ValueError('early_stopping_patience must be a nonnegative integer; 0 disables it')
    if config['evaluation']['k']!=100:
        raise ValueError('This protocol reports Recall@100 and NDCG@100; require k=100')
    if model_config['item_categorical_features']!=spec['categorical_names'] or model_config['item_numeric_features']!=spec['numeric_names']:
        raise ValueError('Requested item features differ from frozen prepared encoding')
    if len(spec['categorical_names'])!=7 or len(spec['numeric_names'])!=4:
        raise ValueError('This release requires all eleven content features')
    device=torch.device(config['training']['device'])
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('Select and connect a Colab GPU runtime before preflight/train')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(config['training']['deterministic'])
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        # Small history attention prioritizes deterministic, portable backward.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    seed_all(config['experiment']['seed'])
    feature_spec=extension(root,manifest) if model_config.get('behavior_content_features',False) else None
    users=pq.read_table(root/'user_vocabulary.parquet')['user_id'].to_pylist()
    model=TwoTower(root,spec,users,model_config,feature_spec).to(device)
    model.data_manifest_sha256=hashlib.sha256((sha(root/'manifest.json')+(sha(root/'features-v2'/'checksums.json') if feature_spec else '')).encode()).hexdigest()
    model.feature_spec=feature_spec
    train=Queries(root,manifest['train_shards'],manifest['history_length'],feature_spec)
    validation=Queries(root,manifest['validation_shards'],manifest['history_length'],feature_spec)
    configure_sampling(model,train,config['training'].get('sampling_correction','none'),config['training'].get('sampling_logq_alpha',1.0),config['training'].get('sampling_logq_smoothing',0.0))
    catalog=pq.read_table(root/'candidates.parquet')
    if catalog['item_id'].to_pylist()!=model.item.raw_ids.cpu().tolist():
        raise ValueError('Candidate/encoding raw IDs differ')
    rule=config['evaluation']['candidate_rule']
    if rule not in ['upload_next_day','first_observed_exposure']:
        raise ValueError('Freeze an explicit candidate eligibility rule')
    name='eligible_after_ms' if rule=='upload_next_day' else 'first_observed_ms'
    eligible=torch.tensor(catalog[name].to_numpy().copy(),device=device)
    return manifest,spec,model,train,validation,eligible,device


def preflight_report(model,train,validation,eligible,device,spec,config):
    _,raw=next(train.batches(32,config['experiment']['seed']))
    batch={n:t.to(device) for n,t in raw.items()}
    report=check(model,batch,spec,config['training']['temperature'])
    ids=validation.values['item_id'];pos=np.searchsorted(model.item.raw_ids.cpu().numpy(),ids)
    allowed=eligible.cpu().numpy()[pos]<=validation.values['time_ms']
    report['sampling_correction']=model.sampling_report
    report['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    report['model_structure']=str(model)
    if config.get('ablation',{}).get('scope')=='architecture':
        assert report['user_mlp_input']==131 and report['item_mlp_input']==64, 'Missing enriched inputs: expected user=131/item=64'
        assert model.sampling_report['mode']=='none', 'Architecture runs require No Correction'
    report.update(device=str(device),cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None,train_queries=train.rows,validation_queries=validation.rows,candidate_rule=config['evaluation']['candidate_rule'],validation_target_coverage=float(allowed.mean()),gpu_feature_acceptance=device.type=='cuda',training_started=False)
    report.update(code_sha256=code_hash(),manifest_sha256=model.data_manifest_sha256,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
    report['limitations']=['Uncovered targets remain in the Recall/NDCG denominator.','first_observed_exposure, when selected, has coverage by construction and is not verified catalog registration.','Unversioned publication attributes use the fixed-content assumption.']
    return report


def update_batch(model,batch,optimizer,scaler,device,amp,temperature,notice=print):
    """Retry the SAME batch on AMP overflow, before any optimizer update.

    Restore RNG and BatchNorm running statistics so retrying doesn't consume
    dropout randomness or update batch statistics twice. Persistent nonfinite
    gradients and all FP32 failures remain hard errors.
    """
    before_rng=rng_state() if amp else None
    running={n:b.clone() for n,b in model.named_buffers() if n.endswith(('running_mean','running_var','num_batches_tracked'))} if amp else {}
    for attempt in range(9):
        if attempt:
            restore_rng(before_rng)
            with torch.no_grad():
                buffers=dict(model.named_buffers())
                for n,value in running.items():buffers[n].copy_(value)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=amp):
            users,items=model(batch)
            loss=loss_fn(users,items,batch['item_id'],temperature,batch_probability(model,batch['item_id']),getattr(model,'sampling_logq_alpha',1.0))
        if not torch.isfinite(loss):raise ValueError('Nonfinite forward loss; no parameter update')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            if not amp or attempt==8:
                raise ValueError(f'Nonfinite gradient after {attempt} AMP retries; no parameter update; parameters={bad[:8]}')
            old=float(scaler.get_scale())
            scaler.update(new_scale=old/2)
            notice(json.dumps({'event':'amp_overflow_retry','retry':attempt+1,'old_scale':old,'new_scale':old/2,'parameter_updates':0,'parameters':bad[:8]}),flush=True)
            continue
        scaler.step(optimizer);scaler.update()
        return loss,len(users),attempt


# A narrowly scoped compatibility migration: v2 checkpoints written before the
# AMP retry fix can resume with identical data/config/runtime/output bindings.
LEGACY_AMP_CODE='035bd69e7cc83d9c53c7072cb89feeba01a83864b30c746500473436428284c4'
LEGACY_HISTORY_DTYPE_CODE='26604fa7a31239f66dfc111ae5a663facc312d2d34c2490107e49dc1e26c5048'


def validate_resume_binding(state,binding,output):
    old=state['binding']
    same_other={k:v for k,v in old.items() if k!='code_sha256'}=={k:v for k,v in binding.items() if k!='code_sha256'}
    permitted=old.get('code_sha256') in [LEGACY_AMP_CODE,LEGACY_HISTORY_DTYPE_CODE] and same_other
    if (old!=binding and not permitted) or state['output']!=str(output):
        raise ValueError('Checkpoint config/data/code/runtime/output binding differs')
    migrations=list(state.get('code_migrations',[]))
    if old!=binding:
        migrations.append({'reason':'AMP overflow retry / history accumulation dtype fixes; architecture unchanged','from':old['code_sha256'],'to':binding['code_sha256'],'checkpoint_step':state['step']})
        print(json.dumps({'checkpoint_code_migration':migrations[-1]}),flush=True)
    return migrations


def train(root,config,output,resume=False,stop_after=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if not resume and any(output.iterdir()):
        raise ValueError('New run requires a new empty output folder; use --resume for latest.pt')
    manifest,spec,model,source,validation,eligible,device=load(root,config)
    report=preflight_report(model,source,validation,eligible,device,spec,config)
    json_write(output/'feature_acceptance.json',report)
    print(json.dumps({'preflight':report}),flush=True)
    settings=config['training'];optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'],betas=tuple(settings['betas']),weight_decay=settings['weight_decay'])
    amp=settings['mixed_precision'] and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    binding={'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'manifest_sha256':model.data_manifest_sha256,'code_sha256':code_hash(),'torch':torch.__version__,'python':platform.python_version(),'device_type':device.type}
    if settings.get('sampling_correction','none')=='logq':
        binding['sampling_counts_sha256']=model.sampling_report['counts_sha256']
    epoch,cursor,step,consumed=0,0,0,0
    best=(-1.0,-1.0);best_epoch=None;run_id=uuid.uuid4().hex[:12]
    stale_epochs=0;stopped_early=False
    epoch_loss_sum=0.0;epoch_rows=0;epoch_train_seconds=0.0
    patience=settings.get('early_stopping_patience',0)
    state=None
    migrations=[]
    if resume:
        state=torch.load(output/'latest.pt',map_location=device,weights_only=False)
        migrations=validate_resume_binding(state,binding,output)
        if state['binding']['code_sha256'] in [LEGACY_AMP_CODE,LEGACY_HISTORY_DTYPE_CODE]:
            backup=output/('latest-before-amp-fix.pt' if state['binding']['code_sha256']==LEGACY_AMP_CODE else 'latest-before-history-dtype-fix.pt')
            if not backup.exists():
                temporary=backup.with_suffix('.pending')
                shutil.copyfile(output/'latest.pt',temporary);temporary.replace(backup)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        epoch,cursor,step,consumed=state['epoch'],state['cursor'],state['step'],state['consumed']
        best,best_epoch,run_id=tuple(state['best']),state['best_epoch'],state['run_id']
        stale_epochs=state.get('stale_epochs',0)
        stopped_early=state.get('stopped_early',False)
        epoch_loss_sum=state.get('epoch_loss_sum',0.0);epoch_rows=state.get('epoch_rows',0)
        epoch_train_seconds=state.get('epoch_train_seconds',0.0)
    tracking=None
    options=config['wandb']
    if options['mode']!='disabled':
        import wandb
        if options['mode']=='online' and not os.environ.get('WANDB_API_KEY'):
            raise ValueError('Add WANDB_API_KEY to Colab Secrets before training')
        tracking=wandb.init(entity=options['entity'],project=options['project'],name=options['name']+('-PROBE' if output.name=='probe' else ''),group=options['group'],id=run_id,resume='must' if resume else 'never',mode=options['mode'],config=config,dir=str(output),tags=options['tags']+(['probe-only'] if output.name=='probe' else []))
        tracking.define_metric('train_step');tracking.define_metric('train/*',step_metric='train_step')
        tracking.define_metric('epoch');tracking.define_metric('validation/*',step_metric='epoch')
        tracking.summary['code_sha256']=binding['code_sha256']
        tracking.summary['purpose']='probe-only' if output.name=='probe' else 'training'
        tracking.summary['sampling_correction']=model.sampling_report
        if migrations:tracking.summary['code_migrations']=migrations
    def upload_epochs():
        before=rng_state()
        try:sync_epoch_artifacts(tracking,output)
        finally:restore_rng(before)
    upload_epochs()
    if state:restore_rng(state['rng'])
    seed=config['experiment']['seed']
    json_write(output/'config.json',config)

    def save(best_now=False,epoch_now=False,write_checkpoint=True):
        current={'binding':binding,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'epoch':epoch,'cursor':cursor,'step':step,'consumed':consumed,'best':best,'best_epoch':best_epoch,'run_id':run_id,'output':str(output),'feature_acceptance':report,'code_migrations':migrations,'stale_epochs':stale_epochs,'stopped_early':stopped_early,'epoch_loss_sum':epoch_loss_sum,'epoch_rows':epoch_rows,'epoch_train_seconds':epoch_train_seconds}
        if write_checkpoint:
            # Epoch/best files precede latest: latest is the durable progress commit.
            if epoch_now:atomic_save(output/f'epoch-{epoch:03d}.pt',current)
            if best_now:atomic_save(output/'best.pt',current)
            atomic_save(output/'latest.pt',current)
        json_write(output/'run_manifest.json',{'binding':binding,'run_id':run_id,'epochs_completed':epoch,'next_batch':cursor,'global_step':step,'consumed':consumed,'target_samples':source.rows*settings['epochs'],'complete':epoch==settings['epochs'] or stopped_early,'stop_reason':'early_stopping' if stopped_early else ('max_epochs' if epoch==settings['epochs'] else None),'stopped_early':stopped_early,'early_stopping_patience':patience,'epochs_without_improvement':stale_epochs,'best_epoch':best_epoch,'best_recall_at_100':best[0],'best_ndcg_at_100':best[1],'latest_checkpoint':str(output/'latest.pt'),'best_checkpoint':str(output/'best.pt') if best_epoch is not None else None,'candidate_rule':config['evaluation']['candidate_rule'],'trainable_parameters':report['trainable_parameters'],'run_url':tracking.url if tracking else None})
        if tracking:
            tracking.summary.update({'epochs_completed':epoch,'best_epoch':best_epoch,'best_recall_at_100':best[0],'best_ndcg_at_100':best[1],'stopped_early':stopped_early,'epochs_without_improvement':stale_epochs})

    def log(metrics):
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
        print(json.dumps(metrics),flush=True)
        if tracking:tracking.log(metrics)

    try:
        if not resume:save()
        elif epoch==settings['epochs'] or stopped_early:save(write_checkpoint=False)
        while epoch<settings['epochs'] and not stopped_early:
            print(json.dumps({'phase':'training','epoch':epoch+1,'step':step}),flush=True)
            model.train();iterator=iter(source.batches(settings['batch_size'],seed+epoch,cursor))
            while True:
                start=time.perf_counter()
                try:next_cursor,raw=next(iterator)
                except StopIteration:break
                batch={n:t.to(device) for n,t in raw.items()}
                loss,batch_rows,retries=update_batch(model,batch,optimizer,scaler,device,amp,settings['temperature'])
                cursor=next_cursor;step+=1;consumed+=batch_rows
                if device.type=='cuda':torch.cuda.synchronize()
                epoch_loss_sum+=float(loss.detach())*batch_rows;epoch_rows+=batch_rows
                epoch_train_seconds+=time.perf_counter()-start
                if step%settings['log_every']==0 or cursor*settings['batch_size']>=source.rows:
                    if device.type=='cuda':torch.cuda.synchronize()
                    metrics={'train_step':step,'train/loss':float(loss.detach()),'train/batch_size':batch_rows,'train/epoch':epoch+1,'train/consumed':consumed,'train/step_seconds':time.perf_counter()-start,'train/amp_retries':retries,'train/loss_scale':float(scaler.get_scale())}
                    if device.type=='cuda':metrics['train/peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
                    log(metrics)
                if step%settings['checkpoint_every']==0:save()
                if stop_after is not None and step>=stop_after:
                    save();return
            evaluation_settings={**config['evaluation'],'output_dim':config['model']['output_dim']}
            print(json.dumps({'phase':'validation','epoch':epoch+1,'queries':validation.rows}),flush=True)
            result=evaluate(model,validation,eligible,evaluation_settings,device,output/f'validation-epoch-{epoch+1:03d}.parquet')
            loss_begin=time.perf_counter()
            result['loss']=validation_loss(model,validation,device,settings['temperature'],256)
            result['validation_loss_seconds']=time.perf_counter()-loss_begin
            result['loss_protocol']='stored-order all validation events; batch=256; no correction; FP32'
            result['train_loss']=epoch_loss_sum/epoch_rows
            result['train_seconds']=epoch_train_seconds
            result['learning_rate']=optimizer.param_groups[0]['lr']
            score=(result['recall_at_100'],result['ndcg_at_100'])
            improved=score>best
            epoch+=1;cursor=0
            if improved:best,best_epoch=score,epoch
            stale_epochs=0 if improved else stale_epochs+1
            stopped_early=bool(patience and stale_epochs>=patience)
            log({'epoch':epoch,'train_step':step,'train/epoch_loss':result['train_loss'],'train/learning_rate':result['learning_rate'],'validation/improved':improved,'validation/epochs_without_improvement':stale_epochs,'validation/stopped_early':stopped_early,**{'validation/'+k:v for k,v in result.items()}})
            json_write(output/f'evaluation-epoch-{epoch:03d}.json',result)
            epoch_loss_sum=0.0;epoch_rows=0;epoch_train_seconds=0.0
            save(improved,epoch_now=True)
            upload_epochs()
            if stopped_early:
                print(json.dumps({'event':'early_stopping','epoch':epoch,'patience':patience,'best_epoch':best_epoch,'best_recall_at_100':best[0],'best_ndcg_at_100':best[1]}),flush=True)
        return output/'best.pt'
    finally:
        if tracking:tracking.finish()


@torch.no_grad()
def export_best(root,config,output):
    output=Path(output).resolve()
    manifest,spec,model,_,_,_,device=load(root,config)
    state=torch.load(output/'best.pt',map_location=device,weights_only=False)
    if state['binding']['manifest_sha256']!=model.data_manifest_sha256 or state['binding']['config_sha256']!=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest() or state['binding']['code_sha256']!=code_hash():
        raise ValueError('Best checkpoint binding differs')
    model.load_state_dict(state['model']);model.eval()
    export=output/'deployment';export.mkdir(exist_ok=True)
    ids=model.item.raw_ids;entries=[]
    for begin in range(0,len(ids),config['evaluation']['item_batch_size']):
        selected=ids[begin:begin+config['evaluation']['item_batch_size']]
        vectors=model.item(selected).cpu().numpy()
        path=export/f'items-{begin:09d}.npz';np.savez(path,item_ids=selected.cpu().numpy(),vectors=vectors)
        entries.append({'path':path.name,'rows':len(selected),'sha256':sha(path)})
    json_write(export/'manifest.json',{'best_epoch':state['best_epoch'],'checkpoint_sha256':sha(output/'best.pt'),'binding':state['binding'],'files':entries,'catalog_size':len(ids),'dimension':config['model']['output_dim'],'note':'Exported deployment artifacts, not a deployed online service. Apply the same candidate eligibility policy at serving time.'})

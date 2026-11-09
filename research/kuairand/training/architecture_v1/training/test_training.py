"""Synthetic correctness only: features, sequence masks, ranks, exact resume."""
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from kt1k.acceptance import check
from kt1k.data import Queries,sha
from kt1k.engine import load,train,export_best,update_batch,validate_resume_binding,LEGACY_AMP_CODE,seed_all
from kt1k.evaluation import ranked_metrics


def fixture(root):
    root.mkdir()
    config=yaml.safe_load(Path(__file__).with_name('configs').joinpath('baseline.yaml').read_text())
    config['training'].update(device='cpu',epochs=2,batch_size=8,mixed_precision=False,log_every=100,checkpoint_every=3)
    config['wandb']['mode']='disabled';config['evaluation'].update(query_batch_size=8,item_batch_size=8)
    cats=config['model']['item_categorical_features'];nums=config['model']['item_numeric_features']
    rng=np.random.default_rng(3)
    data={'item_id':np.arange(16,dtype=np.int64)}
    for n in cats:data['cat_'+n]=np.arange(16,dtype=np.int64)+1
    for n in nums:data['num_'+n]=rng.normal(size=16).astype(np.float32)
    pq.write_table(pa.table(data),root/'item_encoding.parquet')
    pq.write_table(pa.table({'user_id':[0,1,2,3]}),root/'user_vocabulary.parquet')
    pq.write_table(pa.table({'item_id':np.arange(16),'eligible_after_ms':np.zeros(16,dtype=np.int64),'first_observed_ms':np.zeros(16,dtype=np.int64)}),root/'candidates.parquet')
    entries={}
    for split,count in [('train',32),('validation',16)]:
        histories=[[int((i+j)%16) for j in range(1,1+(i%5))] for i in range(count)]
        q={'event_id':np.arange(count)+(0 if split=='train' else 1000),'user_id':np.arange(count)%4,'item_id':np.arange(count)%16,'time_ms':np.arange(count)*1000+1650000000000,'tab':np.arange(count)%4,'history_item_ids':pa.array(histories,type=pa.list_(pa.int64())),'history_max_available_ms':np.zeros(count,dtype=np.int64),'long_view':np.ones(count,dtype=np.int64),'is_random':np.zeros(count,dtype=np.int64)}
        path=root/(split+'-000.parquet');pq.write_table(pa.table(q),path)
        entries[split]=[{'path':path.name,'rows':count}]
    spec={'categorical_names':cats,'numeric_names':nums,'cardinalities':[17]*7}
    (root/'encoding_spec.json').write_text(json.dumps(spec))
    manifest={'schema_version':2,'dataset':'KuaiRand-1K','train_shards':entries['train'],'validation_shards':entries['validation'],'history_length':50}
    (root/'manifest.json').write_text(json.dumps(manifest))
    checks={p.name:sha(p) for p in root.iterdir() if p.is_file()}
    (root/'checksums.json').write_text(json.dumps(checks))
    config['model']['behavior_content_features']=False
    return config


def add_extension(root):
    folder=root/'features-v2';folder.mkdir()
    manifest=json.loads((root/'manifest.json').read_text())
    names=['log_duration','aspect_ratio','portrait','tag_count']
    data={'item_id':np.arange(16),'tag_indices':[[i+1,(i+2)%16+1] for i in range(16)]}
    for i,n in enumerate(names):data['num_'+n]=np.arange(16,dtype=np.float32)/16+i
    pq.write_table(pa.table(data),folder/'item_extra.parquet')
    pq.write_table(pa.table({'value':[str(i) for i in range(16)],'index':np.arange(16)+1}),folder/'tag_vocabulary.parquet')
    for entry in manifest['train_shards']+manifest['validation_shards']:
        table=pq.read_table(root/entry['path']);hist=table['history_item_ids'].to_pylist();now=table['time_ms'].to_pylist()
        result={'event_id':table['event_id'],'user_numeric':[[float((i+j)%5) for j in range(6)] for i in range(len(table))]}
        for name in ['history_time_ms','history_play_ms','history_completion','history_completion_known','history_like']:
            result[name]=[[now[i]-1000*(j+1) if name=='history_time_ms' else float(j%2 if name=='history_like' else (j+1)) for j in range(len(h))] for i,h in enumerate(hist)]
        pq.write_table(pa.table(result),folder/entry['path'])
    spec={'version':'fixture-features-v2','base_manifest_sha256':sha(root/'manifest.json'),
          'user_numeric_names':['count1','count7','play','completion','likes','known'],'user_numeric_mean':[0]*6,'user_numeric_std':[1]*6,
          'history_numeric_names':['age','play','completion','known','like'],'item_numeric_names':names,'tag_cardinality':17,'max_tags':2}
    (folder/'feature_spec.json').write_text(json.dumps(spec))
    (folder/'checksums.json').write_text(json.dumps({p.name:sha(p) for p in folder.iterdir()}))


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    return a==b


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory(prefix='sar-1k-training-test-')
        self.base=Path(self.temp.name);self.root=self.base/'data';self.config=fixture(self.root)
    def tearDown(self):self.temp.cleanup()

    def test_all_architectures_use_features_and_handle_empty_history(self):
        for mode,bn,residual in [('mean',False,0),('mean',True,1),('attention',False,0),('lstm',False,0)]:
            config=copy.deepcopy(self.config);config['model'].update(history_encoder=mode,batch_norm=bn,residual_blocks=residual)
            _,spec,model,source,_,_,_=load(self.root,config)
            _,batch=next(source.batches(32,seed=42))
            report=check(model,batch,spec,.1)
            self.assertTrue(report['passed']);self.assertEqual(model.item.input_dim,60);self.assertEqual(model.user.input_dim,108)
            model.train();empty={k:v[:1] for k,v in batch.items()};empty['history_mask']=torch.zeros_like(empty['history_mask']);empty['history_item_ids']=torch.full_like(empty['history_item_ids'],-1)
            u,v=model(empty);self.assertTrue(torch.isfinite(u).all());self.assertEqual(list(v.shape),[1,64])

    def test_rank_ties_and_uncovered(self):
        scores=torch.tensor([[1.,1.,0.],[2.,3.,4.],[5.,6.,7.]])
        ids=torch.tensor([2,4,8]);targets=torch.tensor([1,0,2]);eligible=torch.tensor([0,0,10]);times=torch.tensor([5,5,5])
        r,h,g,c=ranked_metrics(scores,targets,ids,eligible,times,k=2,block=2)
        self.assertEqual(r.tolist(),[2,2,1]);self.assertEqual(h.tolist(),[True,True,False]);self.assertEqual(c.tolist(),[True,True,False])
        self.assertAlmostEqual(float(g[0]),1/np.log2(3),places=10)

    def test_enriched_features_all_architectures_and_masks(self):
        add_extension(self.root)
        for mode in ['mean','attention','lstm']:
            c=copy.deepcopy(self.config);c['model'].update(behavior_content_features=True,history_encoder=mode)
            _,spec,model,source,_,_,_=load(self.root,c)
            _,batch=next(source.batches(32,seed=42))
            self.assertTrue(check(model,batch,spec,.1)['passed'])
            self.assertEqual(model.item.input_dim,64);self.assertEqual(model.user.input_dim,131)
            model.eval()
            original=model.users(batch)
            changed={k:v.clone() for k,v in batch.items()}
            changed['history_feedback'][~changed['history_mask']]=1000
            self.assertTrue(torch.equal(original,model.users(changed)))
            empty={k:v[:1].clone() for k,v in batch.items()}
            empty['history_item_ids'].fill_(-1);empty['history_mask'].zero_();empty['history_feedback'].zero_()
            self.assertTrue(torch.isfinite(model.users(empty)).all())

    def test_combined_sequence_autocast_forward_backward(self):
        add_extension(self.root)
        device=torch.device(os.environ.get('TEST_SEQUENCE_DEVICE','cpu'))
        for mode in ['attention','lstm']:
            c=copy.deepcopy(self.config)
            c['training']['device']=str(device)
            c['model'].update(behavior_content_features=True,history_encoder=mode,user_hidden=[256,128,64],item_hidden=[128,64],batch_norm=True,residual_blocks=1,dropout=.1)
            _,spec,model,source,_,_,_=load(self.root,c)
            _,batch=next(source.batches(32,seed=42));batch={n:v.to(device) for n,v in batch.items()}
            model.train()
            with torch.autocast(device_type=device.type,enabled=True):
                users,items=model(batch)
                loss=__import__('kt1k.model',fromlist=['loss_fn']).loss_fn(users,items,batch['item_id'],.1)
            loss.backward()
            self.assertEqual(list(users.shape),[32,64]);self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA/cuDNN')
    def test_lstm_cuda_preflight_keeps_rnn_backward_enabled(self):
        """Regression test for cuDNN's eval-mode LSTM backward restriction."""
        add_extension(self.root)
        config=copy.deepcopy(self.config)
        config['training']['device']='cuda'
        config['model'].update(behavior_content_features=True,history_encoder='lstm')
        _,spec,model,source,_,_,_=load(self.root,config)
        _,batch=next(source.batches(32,seed=42))
        report=check(model,{name:value.cuda() for name,value in batch.items()},spec,.1)
        self.assertTrue(report['passed'])

    def test_enriched_resume(self):
        add_extension(self.root);self.config['model']['behavior_content_features']=True
        full=self.base/'enriched-full';split=self.base/'enriched-split'
        train(self.root,self.config,full);train(self.root,self.config,split,stop_after=3);train(self.root,self.config,split,resume=True)
        a=torch.load(full/'latest.pt',weights_only=False);b=torch.load(split/'latest.pt',weights_only=False)
        for key in ['model','optimizer','scaler','rng','epoch','cursor','step','consumed','best','best_epoch']:
            self.assertTrue(equal(a[key],b[key]),key)

    def test_amp_overflow_retries_same_batch_without_double_update(self):
        class Small(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4),torch.nn.Dropout(.2),torch.nn.Linear(4,4))
            def forward(self,b):
                u=torch.nn.functional.normalize(self.net(b['x']).float(),dim=-1)
                return u,torch.nn.functional.normalize(b['x'],dim=-1)
        seed_all(17);model=Small();reference=copy.deepcopy(model)
        batch={'x':torch.randn(8,4),'item_id':torch.arange(8)}
        scaler=torch.amp.GradScaler('cpu',init_scale=256)
        reference_scaler=torch.amp.GradScaler('cpu',init_scale=128)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        ref_optimizer=torch.optim.SGD(reference.parameters(),lr=.01)
        handle=next(model.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')) if scaler.get_scale()>128 else g)
        seed_all(42);loss,rows,retries=update_batch(model,batch,optimizer,scaler,torch.device('cpu'),True,.1,notice=lambda *a,**k:None)
        handle.remove()
        seed_all(42);ref_loss,_,_=update_batch(reference,batch,ref_optimizer,reference_scaler,torch.device('cpu'),True,.1)
        self.assertEqual(retries,1);self.assertEqual(rows,8);self.assertEqual(scaler.get_scale(),128)
        self.assertEqual(float(loss.detach()),float(ref_loss.detach()))
        self.assertTrue(equal(model.state_dict(),reference.state_dict()))
        self.assertEqual(int(model.net[1].num_batches_tracked),1)

    def test_fp32_nonfinite_does_not_update_parameters(self):
        class Small(torch.nn.Module):
            def __init__(self):super().__init__();self.linear=torch.nn.Linear(4,4)
            def forward(self,b):return self.linear(b['x']),b['x']
        model=Small();before=copy.deepcopy(model.state_dict())
        next(model.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')))
        optimizer=torch.optim.Adam(model.parameters());scaler=torch.amp.GradScaler('cpu',enabled=False)
        with self.assertRaisesRegex(ValueError,'Nonfinite gradient'):
            update_batch(model,{'x':torch.randn(8,4),'item_id':torch.arange(8)},optimizer,scaler,torch.device('cpu'),False,.1)
        self.assertTrue(equal(before,model.state_dict()));self.assertFalse(optimizer.state)

    def test_resume_migration_is_restricted_to_known_code_and_same_binding(self):
        binding={'code_sha256':'new-fix','config_sha256':'config','manifest_sha256':'data','python':'3.11','device_type':'cuda'}
        state={'binding':{**binding,'code_sha256':LEGACY_AMP_CODE},'output':'/run','step':1000}
        migrations=validate_resume_binding(state,binding,Path('/run'))
        self.assertEqual(migrations[0]['checkpoint_step'],1000)
        for field in ['config_sha256','manifest_sha256','python','device_type']:
            with self.assertRaises(ValueError):validate_resume_binding(state,{**binding,field:'changed'},Path('/run'))
        with self.assertRaises(ValueError):validate_resume_binding({**state,'binding':{**binding,'code_sha256':'unknown'}},binding,Path('/run'))

    def test_early_stop_reset_best_export_and_terminal_resume(self):
        self.config['training'].update(epochs=10,early_stopping_patience=3)
        scores=[(.5,.2),(.4,.2),(.5,.21),(.5,.21),(.49,.2),(.48,.19)]
        def evaluation(*args):
            epoch=int(args[-1].stem.split('-')[-1]);r,n=scores[epoch-1]
            return {'recall_at_100':r,'ndcg_at_100':n,'queries':16}
        out=self.base/'early'
        with patch('kt1k.engine.evaluate',side_effect=evaluation) as evaluate:
            train(self.root,self.config,out)
            self.assertEqual(evaluate.call_count,6)
            state=torch.load(out/'latest.pt',weights_only=False)
            best=torch.load(out/'best.pt',weights_only=False)
            self.assertEqual(state['epoch'],6);self.assertEqual(state['stale_epochs'],3)
            self.assertTrue(state['stopped_early']);self.assertEqual(best['best_epoch'],3)
            summary=json.loads((out/'run_manifest.json').read_text())
            self.assertTrue(summary['complete']);self.assertEqual(summary['stop_reason'],'early_stopping')
            before=sha(out/'latest.pt')
            train(self.root,self.config,out,resume=True)
            self.assertEqual(evaluate.call_count,6);self.assertEqual(before,sha(out/'latest.pt'))
        export_best(self.root,self.config,out)
        self.assertEqual(json.loads((out/'deployment'/'manifest.json').read_text())['best_epoch'],3)

    def test_early_stop_counter_survives_mid_epoch_resume(self):
        self.config['training'].update(epochs=10,early_stopping_patience=3)
        def evaluation(*args):
            epoch=int(args[-1].stem.split('-')[-1])
            return {'recall_at_100':1/epoch,'ndcg_at_100':1/epoch,'queries':16}
        full=self.base/'early-full';split=self.base/'early-split'
        with patch('kt1k.engine.evaluate',side_effect=evaluation):
            train(self.root,self.config,full)
            train(self.root,self.config,split,stop_after=10)
            interrupted=torch.load(split/'latest.pt',weights_only=False)
            self.assertEqual(interrupted['epoch'],2);self.assertEqual(interrupted['stale_epochs'],1)
            train(self.root,self.config,split,resume=True)
        a=torch.load(full/'latest.pt',weights_only=False);b=torch.load(split/'latest.pt',weights_only=False)
        for key in ['model','optimizer','scaler','rng','epoch','cursor','step','consumed','best','best_epoch','stale_epochs','stopped_early']:
            self.assertTrue(equal(a[key],b[key]),key)
        self.assertEqual(a['epoch'],4);self.assertTrue(a['stopped_early'])

    def test_resume_matches_uninterrupted_and_each_epoch_evaluates(self):
        full=self.base/'full';split=self.base/'split'
        train(self.root,self.config,full)
        train(self.root,self.config,split,stop_after=3)
        train(self.root,self.config,split,resume=True)
        a=torch.load(full/'latest.pt',weights_only=False);b=torch.load(split/'latest.pt',weights_only=False)
        for key in ['model','optimizer','scaler','rng','epoch','cursor','step','consumed','best','best_epoch']:
            self.assertTrue(equal(a[key],b[key]),key)
        self.assertEqual(a['consumed'],64)
        self.assertEqual(len(list(full.glob('evaluation-epoch-*.json'))),2)
        best=torch.load(full/'best.pt',weights_only=False)
        reports=[json.loads(p.read_text()) for p in full.glob('evaluation-epoch-*.json')]
        self.assertEqual(tuple(best['best']),max((x['recall_at_100'],x['ndcg_at_100']) for x in reports))


if __name__=='__main__':unittest.main()

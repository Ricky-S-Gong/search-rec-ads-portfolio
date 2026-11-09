"""Regression checks for architecture scope, epoch recording, and task safety."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from architecture import IDS, config_for, validate_matrix
from kt1k.engine import load, train
from kt1k.model import loss_fn
from kt1k.recording import validation_loss, sync_epoch_artifacts
from test_training import fixture, add_extension, equal


class Stage12Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.data=self.root/'data';self.c=fixture(self.data);add_extension(self.data)
        self.c['model']['behavior_content_features']=True
    def tearDown(self):self.temp.cleanup()

    def test_six_configs_have_only_architecture_changes(self):
        validate_matrix()
        self.assertEqual(len(IDS),6)
        for name in IDS:
            c=config_for(name,2026)
            self.assertEqual(c['experiment']['seed'],2026)
            self.assertEqual(c['training']['sampling_correction'],'none')
            self.assertEqual(c['training']['early_stopping_patience'],3)

    def test_masked_padding_and_max_history_for_all_six(self):
        for name in IDS:
            c=copy.deepcopy(self.c);c['model']=config_for(name,42)['model']
            _,_,model,source,_,_,_=load(self.data,c)
            self.assertEqual((model.user.input_dim,model.item.input_dim),(131,64))
            self.assertFalse(any(isinstance(x,torch.nn.BatchNorm1d) for x in model.modules()))
            if name=='B0':
                self.assertFalse(any(isinstance(x,(torch.nn.MultiheadAttention,torch.nn.LSTM)) for x in model.modules()))
            _,batch=next(source.batches(8))
            for length in [0,1,50]:
                changed={k:v.clone() for k,v in batch.items()}
                changed['history_item_ids'].fill_(-1);changed['history_mask'].zero_();changed['history_feedback'].zero_()
                if length:
                    changed['history_item_ids'][:,-length:]=torch.arange(length)%16
                    changed['history_mask'][:,-length:]=True
                model.eval();u=model.users(changed)
                self.assertTrue(torch.isfinite(u).all())
                self.assertTrue(torch.allclose(u.norm(dim=1),torch.ones(8),atol=1e-5))
                changed['history_feedback'][~changed['history_mask']]=999
                self.assertTrue(torch.equal(u,model.users(changed)))
                if name in ['A1','A3'] and length==50:
                    changed['history_item_ids']=changed['history_item_ids'].flip(1)
                    changed['history_feedback']=changed['history_feedback'].flip(1)
                    self.assertGreater(float((u-model.users(changed)).abs().max()),1e-6)

    def test_fixed_validation_loss_is_uncorrected_and_deterministic(self):
        _,_,model,_,val,_,device=load(self.data,self.c)
        a=validation_loss(model,val,device,.1,8)
        model.sampling_logq_alpha=1.0
        model.sampling_correction='logq'
        b=validation_loss(model,val,device,.1,8)
        self.assertEqual(a,b)
        _,batch=next(val.batches(16))
        u,v=model(batch);self.assertTrue(torch.isfinite(loss_fn(u,v,batch['item_id'],.1)))

    def test_epoch_checkpoints_metrics_and_loss_sums_resume(self):
        full=self.root/'full';split=self.root/'split'
        train(self.data,self.c,full);train(self.data,self.c,split,stop_after=3);train(self.data,self.c,split,resume=True)
        for epoch in [1,2]:
            a=torch.load(full/f'epoch-{epoch:03d}.pt',weights_only=False)
            b=torch.load(split/f'epoch-{epoch:03d}.pt',weights_only=False)
            for key in ['model','optimizer','rng','epoch','cursor','step','epoch_loss_sum','epoch_rows']:
                self.assertTrue(equal(a[key],b[key]),key)
            am=json.loads((full/f'evaluation-epoch-{epoch:03d}.json').read_text())
            bm=json.loads((split/f'evaluation-epoch-{epoch:03d}.json').read_text())
            self.assertEqual(am['train_loss'],bm['train_loss'])
            self.assertEqual(am['loss'],bm['loss'])

    def test_failed_upload_has_no_success_sidecar(self):
        import types
        path=self.root/'epoch-001.pt';path.write_bytes(b'unit-test')
        fake=types.SimpleNamespace(Artifact=lambda *a,**k:types.SimpleNamespace(add_file=lambda *a,**k:None))
        tracking=types.SimpleNamespace(id='unit',log_artifact=lambda a:types.SimpleNamespace(wait=lambda:(_ for _ in ()).throw(RuntimeError('offline'))))
        with patch.dict('sys.modules',{'wandb':fake}):
            with self.assertRaisesRegex(RuntimeError,'offline'):sync_epoch_artifacts(tracking,self.root)
        self.assertFalse(path.with_suffix('.artifact.json').exists())

    def test_resume_repairs_manifest_after_commit_interruption(self):
        from kt1k.engine import json_write
        c=copy.deepcopy(self.c);c['training']['epochs']=1
        out=self.root/'crash'
        def interrupt(path,value):
            if path.name=='run_manifest.json' and value['epochs_completed']==1:
                raise RuntimeError('simulated process exit after checkpoint commit')
            return json_write(path,value)
        with patch('kt1k.engine.json_write',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'simulated process exit'):
                train(self.data,c,out)
        self.assertTrue((out/'epoch-001.pt').exists())
        train(self.data,c,out,resume=True)
        self.assertTrue(json.loads((out/'run_manifest.json').read_text())['complete'])


if __name__=='__main__':unittest.main()

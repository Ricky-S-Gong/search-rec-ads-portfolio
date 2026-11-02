"""Analytical loss/gradient checks and corrected-objective exact resume."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from kt1k.model import loss_fn
from kt1k.engine import load,train,validate_resume_binding
from kt1k.sampling import batch_probability
from test_training import fixture,equal,add_extension
import test_training as original_tests

class LogQTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_analytical_probability_gradient_temperature_and_duplicate(self):
        # Three unequal probabilities distinguish per-column correction from per-row.
        u=torch.tensor([[1.,0.],[0.,1.],[.6,.8]],requires_grad=True)
        v=torch.tensor([[.8,.6],[1.,0.],[.8,.6]],requires_grad=True)
        ids=torch.tensor([10,20,10]);q=torch.tensor([.1,.8,.1]);temp=.2
        loss=loss_fn(u,v,ids,temp,q)
        score=u@v.T/temp
        expected=[]
        for i in range(3):
            candidates=[j for j in range(3) if j==i or ids[j]!=ids[i]]
            terms=torch.stack([score[i,j]-torch.log(q[j]) for j in candidates])
            expected.append(torch.logsumexp(terms,0)-(score[i,i]-torch.log(q[i])))
        manual=torch.stack(expected).mean()
        self.assertTrue(torch.allclose(loss,manual,atol=1e-6))
        actual_grad=torch.autograd.grad(loss,(u,v),retain_graph=True)
        manual_grad=torch.autograd.grad(manual,(u,v))
        for a,b in zip(actual_grad,manual_grad):self.assertTrue(torch.allclose(a,b,atol=1e-6))
        self.assertTrue(torch.allclose(loss_fn(u,v,ids,temp,torch.ones(3)/3),loss_fn(u,v,ids,temp)))
        half=loss_fn(u,v,ids,temp,q,logq_alpha=.5)
        expected_half=[]
        for i in range(3):
            candidates=[j for j in range(3) if j==i or ids[j]!=ids[i]]
            terms=torch.stack([score[i,j]-.5*torch.log(q[j]) for j in candidates])
            expected_half.append(torch.logsumexp(terms,0)-(score[i,i]-.5*torch.log(q[i])))
        self.assertTrue(torch.allclose(half,torch.stack(expected_half).mean(),atol=1e-6))
        for bad in [torch.tensor([0.,.8,.1]),torch.tensor([float('nan'),.8,.1]),torch.tensor([1.1,.8,.1]),torch.ones(3,1)]:
            with self.assertRaises(ValueError):loss_fn(u,v,ids,temp,bad)

    def test_train_only_counts_and_nonpersistent_buffer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'data';config=fixture(root);config['training']['sampling_correction']='logq'
            _,_,model,source,validation,_,_=load(root,config)
            np.testing.assert_allclose(batch_probability(model,torch.arange(16)).numpy(),np.full(16,1/16))
            self.assertEqual(model.sampling_report['train_queries'],32)
            self.assertEqual(model.sampling_report['distinct_targets'],16)
            before=model.sampling_report.copy()
            # Validation feedback has no route into Q; recompute with the same train source.
            validation.values['item_id'][:]=0
            from kt1k.sampling import configure_sampling
            configure_sampling(model,source,'logq')
            self.assertEqual(before,model.sampling_report)
            self.assertNotIn('sampling_probability',model.state_dict())
            source.values['item_id'][:16]=0
            configure_sampling(model,source,'logq')
            self.assertNotEqual(before['counts_sha256'],model.sampling_report['counts_sha256'])
            self.assertAlmostEqual(float(batch_probability(model,torch.tensor([0]))[0]),17/32)
            configure_sampling(model,source,'logq',alpha=.5,smoothing=1.0)
            self.assertEqual(model.sampling_report['alpha'],.5)
            self.assertEqual(model.sampling_report['smoothing'],1.0)
            self.assertAlmostEqual(float(batch_probability(model,torch.tensor([0]))[0]),18/48)
            with self.assertRaises(ValueError):configure_sampling(model,source,'typo')
            with self.assertRaises(ValueError):configure_sampling(model,source,'logq',alpha=1.1)

    def test_corrected_resume_exact_and_old_objective_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'data';config=fixture(root)
            config['training']['sampling_correction']='logq'
            add_extension(root);config['model'].update(behavior_content_features=True,history_encoder='attention',batch_norm=True,residual_blocks=1,dropout=.1)
            full=base/'full';split=base/'split'
            train(root,config,full);train(root,config,split,stop_after=3);train(root,config,split,resume=True)
            a=torch.load(full/'latest.pt',weights_only=False);b=torch.load(split/'latest.pt',weights_only=False)
            for name in ['model','optimizer','rng','step','epoch','consumed','best','stale_epochs']:
                self.assertTrue(equal(a[name],b[name]),name)
            self.assertIn('sampling_counts_sha256',a['binding'])
            old=copy.deepcopy(config);old['training'].pop('sampling_correction')
            with self.assertRaisesRegex(ValueError,'binding differs'):train(root,old,split,resume=True)

class CorrectedEarlyStoppingTests(original_tests.Tests):
    def setUp(self):
        super().setUp()
        self.config['training']['sampling_correction']='logq'

# Reuse just the two meaningful early-stop/resume integration cases in logQ mode.
for _name in dir(original_tests.Tests):
    if _name.startswith('test_') and _name not in ('test_early_stop_reset_best_export_and_terminal_resume','test_early_stop_counter_survives_mid_epoch_resume'):
        setattr(CorrectedEarlyStoppingTests,_name,None)

if __name__=='__main__':unittest.main()

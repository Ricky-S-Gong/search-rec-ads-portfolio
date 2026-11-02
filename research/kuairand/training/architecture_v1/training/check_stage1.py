"""Synthetic enriched-data GPU/CPU smoke, exact resume, and optional online Artifact."""
import argparse
import copy
import json
import os
from pathlib import Path

import torch

from architecture import config_for, validate_matrix
from kt1k.engine import load, preflight_report, train, json_write
from test_training import fixture, add_extension, equal


def check_stage1(output, device='cuda', online=False):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Smoke directory must be new. Reuse prior PASS, or choose a new smoke folder.')
    torch.set_num_threads(1)
    validate_matrix()
    data = output / 'synthetic-data'
    fixture(data); add_extension(data)
    reports = {}
    for name in ['B0','R0','A1','A2','A3','A5']:
        config = config_for(name,42)
        config['training'].update(device=device,batch_size=8,epochs=2,mixed_precision=device=='cuda',checkpoint_every=3)
        config['evaluation'].update(query_batch_size=8,item_batch_size=8)
        config['wandb']['mode']='disabled'
        _,spec,model,source,val,eligible,dev=load(data,config)
        report=preflight_report(model,source,val,eligible,dev,spec,config)
        assert (report['user_mlp_input'],report['item_mlp_input'])==(131,64)
        reports[name]=report
        # Controlled uninterrupted vs interrupted/resumed execution for both controls.
        if name in ['B0','R0']:
            full=output/f'{name}-full';split=output/f'{name}-resumed'
            train(data,config,full)
            train(data,config,split,stop_after=3)
            train(data,config,split,resume=True)
            a=torch.load(full/'latest.pt',map_location='cpu',weights_only=False)
            b=torch.load(split/'latest.pt',map_location='cpu',weights_only=False)
            fields=['model','optimizer','scaler','rng','epoch','cursor','step','consumed','best','best_epoch',
                    'epoch_loss_sum','epoch_rows','stale_epochs']
            for field in fields:
                assert equal(a[field],b[field]), f'{name} resume mismatch: {field}'
            assert len(list(full.glob('epoch-*.pt')))==2
            for epoch in [1,2]:
                af=json.loads((full/f'evaluation-epoch-{epoch:03d}.json').read_text())
                bf=json.loads((split/f'evaluation-epoch-{epoch:03d}.json').read_text())
                for key in ['loss','train_loss','recall_at_100','ndcg_at_100']:
                    assert af[key]==bf[key], (name,key)
    if online:
        if not os.environ.get('WANDB_API_KEY'):raise ValueError('Missing WANDB_API_KEY')
        config=config_for('B0',42)
        config['training'].update(device=device,batch_size=8,epochs=1,mixed_precision=device=='cuda')
        config['evaluation'].update(query_batch_size=8,item_batch_size=8)
        config['wandb'].update(name='SMOKE-ONLY-artifact-check',group='smoke-only')
        train(data,config,output/'artifact-smoke')
        assert len(list((output/'artifact-smoke').glob('epoch-*.artifact.json')))==1
    result={'status':'PASS','purpose':'SYNTHETIC SMOKE ONLY; NOT EXPERIMENT RESULTS',
            'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
            'torch':torch.__version__,'controls_exact_resume':['B0','R0'],
            'all_six_preflight_passed':True,'online_artifact_verified':online,
            'reports':reports}
    json_write(output/'stage1_pass.json',result)
    print('STAGE 1 PASS:',output/'stage1_pass.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--online',action='store_true')
    a=p.parse_args();check_stage1(a.output.resolve(),a.device,a.online)

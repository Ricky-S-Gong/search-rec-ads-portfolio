"""Reproduce the local CPU check only; Colab must use native locked-runtime tests."""
from pathlib import Path
import json
import sys
import unittest
import torch
import numpy
import pyarrow
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
shim=not hasattr(torch.amp,'GradScaler')
if shim:
    torch.amp.GradScaler=lambda device='cuda',enabled=True,**kw:torch.cuda.amp.GradScaler(enabled=False,**kw)
import test_training
if shim:
    test_training.Tests.test_amp_overflow_retries_same_batch_without_double_update=unittest.skip(
        'Local torch lacks CPU GradScaler; native locked-runtime Colab test remains required')(
        test_training.Tests.test_amp_overflow_retries_same_batch_without_double_update)
suite=unittest.defaultTestLoader.loadTestsFromNames(['test_training','test_logq','test_ablation','test_stage12'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
report={'python':sys.version.split()[0],'torch':torch.__version__,'numpy':numpy.__version__,'pyarrow':pyarrow.__version__,
        'device':'CPU','local_scaler_shim':shim,'tests':result.testsRun,'skipped':len(result.skipped),
        'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful(),
        'cuda_verified':False,'online_wandb_verified':False,'full_1k_training_run':False}
Path(__file__).with_name('local_test_result.json').write_text(json.dumps(report,indent=2)+'\n')
sys.exit(not result.wasSuccessful())

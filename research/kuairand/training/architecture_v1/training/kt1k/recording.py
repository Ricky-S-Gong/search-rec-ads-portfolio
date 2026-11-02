"""Common validation loss and recoverable epoch artifacts."""
import json
from pathlib import Path

import torch

from .model import loss_fn
from .data import sha


@torch.no_grad()
def validation_loss(model, queries, device, temperature, batch_size=256):
    """All validation events, fixed stored order/batches, no logQ or AMP."""
    model.eval()
    total, count = 0.0, 0
    for _, raw in queries.batches(batch_size):
        batch = {k: v.to(device) for k, v in raw.items()}
        users, items = model(batch)
        loss = loss_fn(users, items, batch['item_id'], temperature)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite fixed validation loss')
        total += float(loss) * len(users)
        count += len(users)
    if count != queries.rows or not count:
        raise ValueError('Incomplete validation loss traversal')
    return total / count


def sync_epoch_artifacts(tracking, output):
    """Sidecars allow upload retry after a crash without retraining."""
    if tracking is None:
        return
    import wandb
    output = Path(output)
    for checkpoint in sorted(output.glob('epoch-*.pt')):
        sidecar = checkpoint.with_suffix('.artifact.json')
        if sidecar.exists():
            continue
        digest = sha(checkpoint)
        artifact = wandb.Artifact(f'{tracking.id}-{checkpoint.stem}', type='model',
                                  metadata={'sha256': digest, 'epoch': int(checkpoint.stem.split('-')[-1])})
        artifact.add_file(str(checkpoint), name=checkpoint.name)
        uploaded = tracking.log_artifact(artifact)
        uploaded.wait()
        record = {'checkpoint': checkpoint.name, 'sha256': digest,
                  'artifact': uploaded.qualified_name, 'status': 'synced'}
        temporary = sidecar.with_suffix('.pending')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(sidecar)
        print(json.dumps({'event': 'artifact_synced', **record}), flush=True)

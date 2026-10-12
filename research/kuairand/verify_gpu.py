import json
import tempfile
from pathlib import Path

import torch
import yaml

from kuairand.checkpoint import load_checkpoint, save_checkpoint
from kuairand.user_tower import UserTower


def main():
    """Run one bounded CUDA correctness step; never load a real dataset."""
    ROOT = Path(__file__).resolve().parents[2]
    assert torch.cuda.is_available(), "CUDA unavailable in the locked environment"
    config = yaml.safe_load((ROOT / "research/kuairand/configs/week1.yaml").read_text())
    model = UserTower([1, 2, 3], config["model"]).cuda().train()
    features = {
        "user_id": torch.tensor([1, 2, 3, 99], device="cuda"),
        "tab": torch.tensor([0, 1, 2, 99], device="cuda"),
        "time_ms": torch.tensor([1650000000000] * 4, device="cuda"),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        vectors = model(features)
        target = torch.eye(4, 64, device="cuda")
        loss = (vectors - target).square().mean()
    assert vectors.shape == (4, 64) and torch.isfinite(vectors).all()
    assert torch.allclose(vectors.norm(dim=1), torch.ones(4, device="cuda"), atol=1e-4)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    scaler.step(optimizer)
    scaler.update()
    with tempfile.TemporaryDirectory() as directory:
        binding = {"purpose": "bounded_cuda_correctness_check"}
        path = save_checkpoint(
            directory,
            {
                "schema_version": 1,
                "binding": binding,
                "global_step": 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            },
        )
        saved = load_checkpoint(path, binding)
        restored = UserTower([1, 2, 3], config["model"]).cuda()
        restored.load_state_dict(saved["model"])
        assert all(
            torch.equal(value, restored.state_dict()[key])
            for key, value in model.state_dict().items()
        )
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "cuda_verification": "passed",
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                "checks": [
                    "user_tower_forward",
                    "normalized_vectors",
                    "AMP_backward",
                    "Adam_update",
                    "checkpoint_roundtrip",
                ],
                "scope": "one synthetic correctness step, no real experiment",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

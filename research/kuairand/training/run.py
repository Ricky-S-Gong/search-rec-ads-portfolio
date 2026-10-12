"""Unified entry point; training and remote writes require explicit commands."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_config(path: Path) -> tuple[dict, str]:
    """Read a YAML mapping and return its canonical SHA-256; reject invalid basics."""
    import yaml

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    for section in ("experiment", "model", "training", "wandb"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping: {section}")

    def positive_int(value, label):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")

    model = config["model"]
    for field in ("user_embedding_dim", "context_embedding_dim", "num_tabs"):
        positive_int(model.get(field), field)
    if not isinstance(model.get("item_factory"), str) or ":" not in model["item_factory"]:
        raise ValueError("Set item_factory to Ziqi's module:function")
    if not isinstance(model.get("hidden_dims"), list):
        raise ValueError("hidden_dims must be a list")
    for width in model["hidden_dims"]:
        positive_int(width, "hidden dimension")
    if type(model.get("dropout")) not in (int, float) or not 0 <= model["dropout"] < 1:
        raise ValueError("dropout must be in [0,1)")
    if type(model.get("batch_norm")) is not bool:
        raise ValueError("batch_norm must be boolean")
    if type(model.get("timezone_offset_minutes")) is not int:
        raise ValueError("timezone_offset_minutes must be integer; verify with data publisher")
    if config["model"].get("output_dim") != 64:
        raise ValueError("The shared tower contract requires output_dim=64")
    if model.get("history_pooling", "none") not in ("none", "mean"):
        raise ValueError("history_pooling must be none or mean")
    if "item_categorical_embedding_dim" in model:
        positive_int(model["item_categorical_embedding_dim"], "item_categorical_embedding_dim")
    training = config["training"]
    positive_int(training.get("cache_row_groups"), "cache_row_groups")
    positive_int(config.get("export", {}).get("batch_size"), "export.batch_size")
    if training.get("device") not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    for field in ("mixed_precision", "deterministic"):
        if type(training.get(field)) is not bool:
            raise ValueError(f"{field} must be boolean")
    betas = training.get("betas")
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(type(x) not in (int, float) or not 0 <= x < 1 for x in betas)
    ):
        raise ValueError("Adam betas must be two numbers in [0,1)")
    weight_decay = training.get("weight_decay")
    if type(weight_decay) not in (int, float) or not 0 <= weight_decay < float("inf"):
        raise ValueError("weight_decay must be finite and nonnegative")
    if type(config["experiment"].get("seed")) is not int or config["experiment"]["seed"] < 0:
        raise ValueError("seed must be nonnegative integer")
    if training.get("optimizer") != "adam":
        raise ValueError("Week 1 optimizer must be adam")
    for field in ("batch_size", "epochs", "log_every_steps", "checkpoint_every_steps"):
        value = training.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("learning_rate", "temperature"):
        value = training.get(field)
        if type(value) not in (int, float) or not 0 < value < float("inf"):
            raise ValueError(f"{field} must be finite and positive")
    if config["wandb"].get("entity") != "SAR-Lab":
        raise ValueError("Runs must belong to SAR-Lab")
    if not config["wandb"].get("project"):
        raise ValueError("An explicit W&B project is required")
    if config["wandb"].get("mode") not in ("online", "disabled"):
        raise ValueError("Use online tracking or disabled for correctness checks")
    canonical = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    return config, hashlib.sha256(canonical).hexdigest()


def check_env() -> None:
    """Print package/runtime versions and CUDA availability without starting a run."""
    from importlib.metadata import version

    import torch

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("torch", "numpy", "pyarrow", "wandb", "pyyaml")},
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(result, indent=2))


def init_wandb(config: dict, config_hash: str) -> None:
    """Explicit, manually invoked connection check; creates one remote setup run."""
    import wandb
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("Set your own WANDB_API_KEY in .env or Colab Secrets")
    tracking = config["wandb"]
    with wandb.init(
        entity=tracking["entity"],
        project=tracking["project"],
        name=f"{tracking['name']}-setup",
        group=tracking["group"],
        job_type="setup-check",
        tags=tracking["tags"],
        config={"experiment": config["experiment"], "config_sha256": config_hash},
    ) as run:
        run.log({"setup/connection_ok": 1})
        print(f"Run ID: {run.id}\nRun URL: {run.url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "check-env",
            "check-config",
            "init-wandb",
            "prepare-pure",
            "check-data",
            "preflight",
            "train",
            "export",
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "research/kuairand/training/configs/week1.yaml"
    )
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume", type=Path)
    recovery.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--start-id", type=int)
    parser.add_argument("--end-id", type=int)
    args = parser.parse_args()
    try:
        if args.command == "check-env":
            check_env()
            return
        config, digest = load_config(args.config)
        if args.command == "prepare-pure":
            from recommender.prepare_pure import prepare_pure

            if args.source is None or args.output is None:
                parser.error("prepare-pure requires --source and a new --output directory")
            print(json.dumps(prepare_pure(args.source, args.output, config), indent=2))
        elif args.command in ("check-data", "preflight"):
            from recommender.prepare_pure import check_data

            if args.manifest is None:
                parser.error("check-data/preflight requires --manifest")
            check_data(args.manifest, config, preflight=args.command == "preflight")
        elif args.command == "check-config":
            print(json.dumps({"config": config, "sha256": digest}, indent=2))
        elif args.command == "init-wandb":
            init_wandb(config, digest)
        elif args.command == "train":
            from dotenv import load_dotenv

            from recommender.checkpoint import latest_checkpoint
            from recommender.training import train

            load_dotenv(ROOT / ".env", override=False)
            if args.manifest is None or args.output is None:
                parser.error("train requires --manifest and persistent --output")
            if config["wandb"]["mode"] == "online" and not os.getenv("WANDB_API_KEY"):
                parser.error("Set WANDB_API_KEY in .env or Colab Secrets")
            checkpoint = (
                latest_checkpoint(args.output / "checkpoints")
                if args.resume_latest
                else args.resume
            )
            print(train(config, args.manifest, args.output, checkpoint))
        else:
            from recommender.export import export_items

            if any(
                x is None
                for x in (args.manifest, args.checkpoint, args.output, args.start_id, args.end_id)
            ):
                parser.error("export requires manifest/checkpoint/output/start-id/end-id")
            print(
                json.dumps(
                    export_items(
                        config,
                        args.manifest,
                        args.checkpoint,
                        args.output,
                        args.start_id,
                        args.end_id,
                    ),
                    indent=2,
                )
            )
    except (ValueError, OSError, ImportError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

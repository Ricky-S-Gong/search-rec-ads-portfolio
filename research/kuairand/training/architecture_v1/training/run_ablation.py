"""Materialize and run the frozen 1K ablation matrix without touching sealed test data."""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).parent


def merge(target, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def load_plan(path=ROOT / 'configs' / 'ablation_plan.yaml'):
    plan = yaml.safe_load(Path(path).read_text())
    base = yaml.safe_load((Path(path).parent / plan['base_config']).read_text())
    return base, plan['groups']


def materialize(base, group, variant, override, seed):
    config = copy.deepcopy(base)
    merge(config, override)
    config['experiment']['seed'] = seed
    config['ablation'] = {'group': group, 'variant': variant, 'screening_seed': 42}
    wandb = config['wandb']
    wandb['name'] = f'1k-ablation-{variant}-seed{seed}'
    wandb['group'] = f'1k-ablation-{group}'
    wandb['tags'] = list(dict.fromkeys([*wandb.get('tags', []), 'ablation', group, variant, f'seed-{seed}']))
    return config


def selected(groups, requested):
    if requested == 'all':
        return [(group, variant, override) for group, variants in groups.items() for variant, override in variants.items()]
    return [(requested, variant, override) for variant, override in groups[requested].items()]


def write_json(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def choose_variants(groups, group, names):
    variants = selected(groups, group)
    if not names:
        return variants
    requested = set(names.split(','))
    found = {name for _, name, _ in variants}
    unknown = requested - found
    if unknown:
        raise ValueError(f'Unknown variant(s) for {group}: {sorted(unknown)}')
    return [row for row in variants if row[1] in requested]


def main():
    base, groups = load_plan()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--group', choices=[*groups, 'all'], default='all')
    parser.add_argument('--seeds', default='42', help='Comma-separated seeds; use 42 for screening, then 42,2026,3407 for finalists.')
    parser.add_argument('--variants', help='Comma-separated variants for one group; used for finalist reruns.')
    parser.add_argument('--control-only', action='store_true', help='Run only the matched control, for a finalist selection where no variant beat it.')
    parser.add_argument('--preflight', action='store_true', help='Run zero-update GPU checks for every selected configuration.')
    parser.add_argument('--execute', action='store_true', help='Run GPU training. Without this flag, only write resolved configs.')
    parser.add_argument('--continue-on-error', action='store_true', help='Record failed runs and continue with unrelated configurations.')
    args = parser.parse_args()
    if args.preflight and args.execute:
        raise ValueError('Choose either --preflight or --execute')
    if args.variants and args.group == 'all':
        raise ValueError('--variants requires one explicit --group')
    if args.control_only and (args.group == 'all' or args.variants):
        raise ValueError('--control-only requires one explicit --group and no --variants')
    if 'sealed' in str(args.data).lower():
        raise ValueError('Ablation selection must use the train/validation release, never final-test-sealed.')
    seeds = [int(seed) for seed in args.seeds.split(',')]
    variants = [('control', 'control', {})] if args.control_only else [('control', 'control', {})] + choose_variants(groups, args.group, args.variants)
    failures = []
    for group, variant, override in variants:
        for seed in seeds:
            output = args.output_root / group / variant / f'seed-{seed}'
            output.mkdir(parents=True, exist_ok=True)
            config_path = output / 'config.yaml'
            config = materialize(base, group, variant, override, seed)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            run = output / 'run'
            manifest_path = run / 'run_manifest.json'
            status_path = output / ('preflight_status.json' if args.preflight else 'scheduler_status.json')
            label = f'{group}/{variant}/seed-{seed}'
            if args.preflight:
                print(f'PREFLIGHT {label}: {config_path}')
                command = [sys.executable, 'run.py', 'preflight', '--data', str(args.data), '--config', str(config_path), '--output', str(output / 'preflight')]
            elif args.execute:
                if manifest_path.exists() and json.loads(manifest_path.read_text()).get('complete'):
                    write_json(status_path, {'state': 'skipped_complete', 'updated_at': time.time(), 'label': label})
                    print(f'SKIP {label}: completed run exists')
                    continue
                accepted = output / 'preflight' / 'feature_acceptance.json'
                if not accepted.exists() or not json.loads(accepted.read_text()).get('passed'):
                    raise RuntimeError(f'Missing successful preflight for {label}; run --preflight first')
                command = [sys.executable, 'run.py', 'train', '--data', str(args.data), '--config', str(config_path), '--output', str(run)]
                if (run / 'latest.pt').exists():
                    command.append('--resume')
                print(f'RUN {label}: {config_path}')
            else:
                print(f'PLAN {label}: {config_path}')
                continue
            write_json(status_path, {'state': 'running', 'updated_at': time.time(), 'label': label, 'command': command})
            try:
                completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
            except subprocess.CalledProcessError as error:
                stdout = error.stdout or ''
                stderr = error.stderr or ''
                failure = {'state': 'failed', 'updated_at': time.time(), 'label': label, 'returncode': error.returncode, 'command': command,
                           'stdout_tail': stdout[-12000:], 'stderr_tail': stderr[-12000:]}
                write_json(status_path, failure); failures.append(failure)
                if stdout: print(stdout, end='')
                if stderr: print(stderr, end='', file=sys.stderr)
                print(f'FAILED {label}: exit {error.returncode}', file=sys.stderr)
                if not args.continue_on_error:
                    raise
            else:
                if completed.stdout: print(completed.stdout, end='')
                if completed.stderr: print(completed.stderr, end='', file=sys.stderr)
                write_json(status_path, {'state': 'passed' if args.preflight else 'completed', 'updated_at': time.time(), 'label': label, 'command': command})
    summary = {'failures': failures, 'complete': not failures, 'updated_at': time.time()}
    write_json(args.output_root / ('preflight_summary.json' if args.preflight else 'scheduler_summary.json'), summary)
    if failures and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == '__main__':
    raise SystemExit('Historical matrix disabled in architecture-v1. Use architecture.py; see ../README_FIRST_中文.md')

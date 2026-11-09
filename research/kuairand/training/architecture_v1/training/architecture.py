"""Explicit architecture tasks. One GPU process; immutable configs and live logs."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
IDS = ['B0', 'R0', 'A1', 'A2', 'A3', 'A5']


def digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def config_for(name, seed):
    config = yaml.safe_load((ROOT / 'configs' / f'{name.lower()}.yaml').read_text())
    config['experiment']['seed'] = seed
    config['wandb']['name'] = f'architecture-v1-{name}-seed{seed}'
    return config


def validate_matrix():
    baseline = config_for('B0', 42)
    for name in IDS:
        c = config_for(name, 42)
        assert c['model']['batch_norm'] is False
        assert c['model']['residual_blocks'] == c['model']['dropout'] == 0
        assert c['model']['behavior_content_features'] is True
        assert c['model']['output_dim'] == 64 and c['model']['history_length'] == 50
        assert c['training'] == baseline['training']
        assert c['evaluation'] == baseline['evaluation']
        for field in set(c['model']) - {'user_hidden','item_hidden','history_encoder'}:
            assert c['model'][field] == baseline['model'][field], (name, field)
    print('MATRIX PASS: six configs, 18 config/seed tasks; A4 reuses B0', flush=True)


def stream(command, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a', buffering=1) as out:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                out.write(line)
            if process.wait():
                raise RuntimeError(f'Command failed; see {logfile}')
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise


def write_json(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run_tasks(args):
    from kt1k.engine import code_hash
    import torch
    if not torch.cuda.is_available():
        raise ValueError('Connect a GPU runtime first')
    validate_matrix()
    for name in args.ids:
        for seed in args.seeds:
            folder = args.output_root / name / f'seed-{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            lock = folder / 'task.lock'
            try:
                lock.mkdir()
            except FileExistsError:
                raise RuntimeError(f'Task occupied: {lock}. If the old process is DEAD, use the notebook unlock cell.')
            write_json(lock / 'owner.json', {'pid': os.getpid(), 'host': platform.node(), 'started': time.time()})
            try:
                config = config_for(name, seed)
                config_path = folder / 'config.yaml'
                if config_path.exists() and yaml.safe_load(config_path.read_text()) != config:
                    raise ValueError('Existing resolved config differs; use a new experiment root')
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                check_dir = folder / 'preflight'
                if args.command == 'preflight':
                    command = [sys.executable, '-u', 'run.py', 'preflight', '--data', str(args.data),
                               '--config', str(config_path), '--output', str(check_dir)]
                else:
                    accepted = json.loads((check_dir / 'feature_acceptance.json').read_text())
                    expected = {'code_sha256': code_hash(), 'config_sha256': digest(config),
                                'cuda_device': torch.cuda.get_device_name(), 'gpu_feature_acceptance': True,
                                'user_mlp_input': 131, 'item_mlp_input': 64, 'passed': True}
                    if any(accepted.get(k) != v for k, v in expected.items()):
                        raise ValueError('Preflight no longer matches config/code/GPU; rerun preflight')
                    # Train re-verifies all dataset checksums and binding before any update.
                    run = folder / ('probe' if args.command == 'probe' else 'run')
                    command = [sys.executable, '-u', 'run.py', 'train', '--data', str(args.data),
                               '--config', str(config_path), '--output', str(run)]
                    if args.command == 'probe' and (run/'run_manifest.json').exists():
                        prior=json.loads((run/'run_manifest.json').read_text())
                        if prior['binding']['config_sha256']!=digest(config) or prior['binding']['code_sha256']!=code_hash() or prior['binding']['manifest_sha256']!=accepted['manifest_sha256']:
                            raise ValueError('Probe binding differs')
                        if prior['global_step']>=args.steps:
                            print(f'SKIP completed probe: {name} seed {seed}',flush=True)
                            continue
                    if args.command == 'probe':
                        command += ['--stop-after', str(args.steps)]
                    if (run / 'latest.pt').exists():
                        command.append('--resume')
                write_json(folder / f'{args.command}_status.json', {'status':'running','command':command})
                try:
                    stream(command, folder / f'{args.command}.log')
                except BaseException as error:
                    write_json(folder / f'{args.command}_status.json', {'status':'failed','error':str(error)})
                    raise
                write_json(folder / f'{args.command}_status.json', {'status':'passed' if args.command != 'train' else 'completed'})
            finally:
                (lock / 'owner.json').unlink(missing_ok=True)
                lock.rmdir()


def summarize(root):
    rows = []
    for name in IDS:
        for seed in [42, 2026, 3407]:
            folder = root / name / f'seed-{seed}'
            run = folder / 'run'
            row = {'config_id':name,'seed':seed,'status':'pending'}
            path = run / 'run_manifest.json'
            if path.exists():
                m = json.loads(path.read_text())
                row.update(status='completed' if m['complete'] else 'incomplete', **m['binding'])
                for key in ['run_id','run_url','best_epoch','epochs_completed','stop_reason',
                            'best_recall_at_100','best_ndcg_at_100','trainable_parameters','best_checkpoint']:
                    row[key] = m.get(key)
                epoch = m.get('best_epoch')
                if epoch:
                    row.update(json.loads((run / f'evaluation-epoch-{epoch:03d}.json').read_text()))
                epochs=[json.loads(p.read_text()) for p in sorted(run.glob('evaluation-epoch-*.json'))]
                row['total_train_seconds']=sum(x.get('train_seconds',0) for x in epochs)
                row['total_validation_seconds']=sum(x.get('evaluation_seconds',0)+x.get('validation_loss_seconds',0) for x in epochs)
                if (run/'metrics.jsonl').exists():
                    metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
                    row['peak_gpu_memory_bytes']=max((x.get('train/peak_gpu_memory_bytes',0) for x in metrics),default=0)
                row['epochs_saved'] = len(list(run.glob('epoch-*.pt')))
                row['artifacts_synced'] = len(list(run.glob('epoch-*.artifact.json')))
            else:
                status = folder / 'train_status.json'
                if status.exists():
                    row['status'] = json.loads(status.read_text())['status']
            rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root / 'architecture_raw_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write_json(root / 'architecture_results.json', {'rows':rows,'aliases':{'A4':'B0'}})
    for row in rows[:]:
        if row['seed']==42:
            print(row['config_id'], row['status'], row.get('best_epoch'), row.get('best_recall_at_100'))
    return rows


def feedback_bundle(root):
    """Collect small aggregate evidence; exclude data, secrets and model weights."""
    import zipfile
    summarize(root)
    target = root / 'SEND_TO_CHATGPT.zip'
    selected = [root / 'architecture_results.json', root / 'architecture_raw_results.csv',
                root/'stage1_tests.log',root/'environment.json',root/'data_audit.json',root/'resource_budget.json',root/'smoke'/'stage1_pass.json']
    selected += list(root.glob('smoke*/stage1_pass.json'))+list(root.glob('smoke*/artifact-smoke/*.json'))
    selected += list(root.glob('*console.log'))+list(root.glob('stage1_smoke.log'))
    for folder in root.glob('*/seed-*'):
        selected += list(folder.glob('*.log')) + list(folder.glob('*status.json')) + list(folder.glob('config.yaml'))
        selected += list(folder.glob('preflight/*.json'))
        for kind in ['run','probe']:
            sub = folder / kind
            selected += [p for p in sub.glob('*.json') if p.name != 'config.json']
            selected += list(sub.glob('metrics.jsonl'))
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(selected)):
            if path.is_file():
                z.write(path, path.relative_to(root))
    print('SEND THIS FILE:', target)
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['matrix','preflight','probe','train','summary','feedback'])
    p.add_argument('--data', type=Path)
    p.add_argument('--output-root', type=Path, default=Path('runs/architecture-v1'))
    p.add_argument('--ids', nargs='+', choices=IDS, default=['B0','R0'])
    p.add_argument('--seeds', type=int, nargs='+', choices=[42,2026,3407], default=[42])
    p.add_argument('--steps', type=int, default=200)
    args = p.parse_args()
    args.output_root = args.output_root.resolve()
    if args.command == 'matrix': validate_matrix()
    elif args.command == 'summary': summarize(args.output_root)
    elif args.command == 'feedback': feedback_bundle(args.output_root)
    else:
        if args.data is None: p.error('--data is required')
        run_tasks(args)

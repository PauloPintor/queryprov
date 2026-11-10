"""Portable configuration and isolated execution for the scalar evaluator."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys

from .cli import SCALES, dump, sha


def configure(config):
    from sklearn import set_config
    from . import psd as e
    e.INPUT_ROOT, e.OUTPUT_ROOT = Path(config['queries_root']), Path(config['output_dir'])
    e.ANOMALY_CLASSES = (config['scenario'].upper(),)
    e.SPLIT_DIRS = {config['scenario'].upper(): Path(config['splits_dir'])}
    e.QUERY_IDS = tuple(config['queries'])
    e.SCALES = tuple(s for s in SCALES if s['name'] in config['scales'])
    e.INCLUDE_NROWS = config['variant'] == 'psd-r'
    e.SCALAR_FEATURE_NAMES = e.STORED_SCALAR_FEATURE_NAMES if e.INCLUDE_NROWS else e.STORED_SCALAR_FEATURE_NAMES[1:]
    e.SCALAR_DIM = len(e.SCALAR_FEATURE_NAMES)
    e.KNN_K, e.N_JOBS = config['neighbors'], config['jobs']
    e.UNSUP_METHODS = ('knn_distance',)
    set_config(working_memory=config['working_memory_mb'])
    return e


def main(argv=None):
    p = argparse.ArgumentParser(description='PSD (7) or PSD+R (8); StandardScaler fitted on FIT; cosine k-NN')
    for name in ('queries-root', 'splits-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--scenario', choices=['s1', 's2'], required=True)
    p.add_argument('--variant', choices=['psd', 'psd-r'], required=True)
    p.add_argument('--queries', type=int, nargs='+', required=True)
    p.add_argument('--scales', choices=[s['name'] for s in SCALES], nargs='+', default=[s['name'] for s in SCALES])
    p.add_argument('--neighbors', type=int, choices=[1, 5], required=True)
    p.add_argument('--jobs', type=int, default=-1)
    p.add_argument('--working-memory-mb', type=float, default=512)
    p.add_argument('--timeout-seconds', type=float, default=18000)
    p.add_argument('--check-splits', action='store_true')
    p.add_argument('--rerun-completed', action='store_true')
    p.add_argument('--rerun-resource-failures', action='store_true')
    p.add_argument('--_worker-json', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.jobs == 0 or min(a.queries) <= 0 or any(not math.isfinite(v) or v <= 0 for v in (a.working_memory_mb, a.timeout_seconds)):
        p.error('Query IDs and resource limits must be positive; jobs cannot be zero')
    for name in ('queries', 'scales'):
        if len(set(getattr(a, name))) != len(getattr(a, name)):
            p.error(f'Duplicate {name}')
    for name in ('queries_root', 'splits_dir', 'output_dir'):
        setattr(a, name, getattr(a, name).resolve())
    for source in (a.queries_root, a.splits_dir):
        if not source.is_dir():
            p.error(f'Input directory missing: {source}')
        if source == a.output_dir or source in a.output_dir.parents or a.output_dir in source.parents:
            p.error('Output must be separate from input trees')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
              if k not in ('_worker_json', 'check_splits', 'rerun_completed', 'rerun_resource_failures')}
    e = configure(config)
    if a._worker_json:
        return e.worker(a._worker_json)
    jobs = e.preflight()
    code = {f.name: sha(f) for f in Path(__file__).parent.glob('*.py')}
    for request in jobs:
        request['cli_config'] = config
        request['job_signature'] = e.digest_object({'source_signature': request['job_signature'], 'config': config, 'code': code})
    if a.check_splits:
        for request in jobs:
            manifest = e.read_manifest(request['shared_split_file'], request['query_id'], request['scale'])
            for pool, parts in [('normal', ('fit', 'calib', 'test_normal')), ('anomaly', ('test_anom',))]:
                entries = [r for part in parts for r in manifest[part]]
                e.load_pool(Path(request[pool + '_directory']), entries, request[pool + '_input_fingerprint'])
        print(f'[CHECK OK] {len(jobs)} configurations; identities/features checked; no scoring or split writes')
        return 0
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.output_dir / '.evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        context = {'configuration': config, 'code': code, 'jobs': [j['job_signature'] for j in jobs]}
        manifest_path = a.output_dir / 'evaluation_run.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != context:
                raise ValueError('Output configuration, code or inputs changed; use a new output directory')
        elif any(a.output_dir.rglob('*.csv')) or any(a.output_dir.rglob('*.npz')):
            raise ValueError('Unversioned results exist; use a new output directory')
        dump(manifest_path, context)
        failures = 0
        public = list(sys.argv[1:] if argv is None else argv)
        for request in jobs:
            checkpoint = Path(request['result_file'])
            old = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            valid = old.get('job_signature') == request['job_signature']
            score = Path(request['scores_file'])
            if valid and old.get('status') == 'ok' and not a.rerun_completed:
                if not score.is_file() or sha(score) != old.get('scores_sha256'):
                    raise ValueError(f'Saved scores missing/changed: {score}; use --rerun-completed')
                print(f'[REUSE] {checkpoint.stem}', flush=True)
                continue
            if valid and old.get('status') in ('timeout', 'oom_or_sigkill', 'memory_error') and not a.rerun_resource_failures:
                failures += 1
                continue
            checkpoint.unlink(missing_ok=True)
            score.unlink(missing_ok=True)
            Path(request['identity_audit_file']).unlink(missing_ok=True)
            e.write_json(request['request_file'], request)
            print(f'[START] {checkpoint.stem}; {a.variant}; {a.neighbors}-NN', flush=True)
            command = [sys.executable, str(Path(sys.argv[0]).resolve()), *public, '--_worker-json', request['request_file']]
            try:
                result = subprocess.run(command, timeout=a.timeout_seconds, check=False)
                status = 'oom_or_sigkill' if result.returncode == -9 else 'worker_exit'
            except subprocess.TimeoutExpired:
                status = 'timeout'
            if status == 'timeout' or not checkpoint.exists():
                e.write_json(checkpoint, {**request, 'query': f"q{request['query_id']}", 'scale': request['scale']['name'],
                    'status': status, 'comparison_ready': False, 'error': status + '; inspect worker log/OS'})
            row = json.loads(checkpoint.read_text())
            if row.get('error_type') == 'MemoryError':
                row['status'] = 'memory_error'
                e.write_json(checkpoint, row)
            if row.get('status') != 'ok':
                score.unlink(missing_ok=True)
            failures += row.get('status') != 'ok'
            e.aggregate(jobs)
        e.aggregate(jobs)
        dump(a.output_dir / 'evaluation_status.json', {'complete': not failures, 'configurations': len(jobs), 'failures': failures})
        print(f'[DONE] configurations={len(jobs)} failures={failures}', flush=True)
        return int(bool(failures))

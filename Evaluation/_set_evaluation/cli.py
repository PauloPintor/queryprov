"""Portable orchestration; scientific calculations remain in the source engines."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SCALES = [
    {'name': '5k', 'train_k': 5000, 'test_normal_k': 1250, 'n_anom_test': 66},
    {'name': '10k', 'train_k': 10000, 'test_normal_k': 2500, 'n_anom_test': 125},
    {'name': '25k', 'train_k': 25000, 'test_normal_k': 6250, 'n_anom_test': 313},
    {'name': '50k', 'train_k': 50000, 'test_normal_k': 12500, 'n_anom_test': 625},
]
METHODS = {'lineage': 'lineage_jaccard', 'atomic': 'atomic_why_jaccard'}


def methods_for(kind):
    if kind == 'bag_full':
        return {'token': 'token_bag', 'witness': 'witness_bag'}
    return {'lineage': 'vec_lineage', 'atomic': 'vec_atomic'} if kind == 'vec_full' else METHODS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def parse_args(kind, argv=None):
    p = argparse.ArgumentParser(description=f'Evaluate {kind} using existing S1/S2 splits; never create splits.')
    p.add_argument('--queries-root', type=Path, required=True)
    p.add_argument('--splits-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--scenario', choices=['s1', 's2'], required=True)
    p.add_argument('--queries', nargs='+', type=int, required=True)
    p.add_argument('--scales', nargs='+', choices=[s['name'] for s in SCALES], default=[s['name'] for s in SCALES])
    p.add_argument('--representations', nargs='+', choices=list(methods_for(kind)), default=list(methods_for(kind)))
    p.add_argument('--neighbors', type=int, choices=[1, 5], required=True)
    if kind in ('vec_full', 'bag_full'):
        p.add_argument('--jobs', type=int, default=-1, help='Pairwise cosine n_jobs')
        p.add_argument('--working-memory-mb', type=float, default=512, help='Pairwise distance chunk target, not an RSS cap')
    if kind == 'kmv':
        p.add_argument('--kmv-k', type=int, nargs='+', choices=[512, 1024], required=True)
    p.add_argument('--timeout-seconds', type=float, default=18000)
    if kind not in ('vec_full', 'bag_full'):
        p.add_argument('--threads', type=int, help='Numba threads; omitted uses the installed runtime default')
    else:
        p.set_defaults(threads=None)
    p.add_argument('--check-splits', action='store_true', help='Validate existing identities/counts without scoring')
    p.add_argument('--rerun-completed', action='store_true')
    p.add_argument('--rerun-resource-failures', action='store_true')
    p.add_argument('--_worker-json', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if not math.isfinite(a.timeout_seconds) or a.timeout_seconds <= 0 or any(q <= 0 for q in a.queries):
        p.error('Timeout and query IDs must be positive')
    if a.threads is not None and a.threads < 1:
        p.error('--threads must be positive')
    if kind in ('vec_full', 'bag_full') and (a.jobs == 0 or not math.isfinite(a.working_memory_mb) or a.working_memory_mb <= 0):
        p.error('jobs cannot be zero; working memory must be positive and finite')
    for name in ('queries', 'scales', 'representations') + (('kmv_k',) if kind == 'kmv' else ()):
        if len(set(getattr(a, name))) != len(getattr(a, name)):
            p.error(f'Duplicate values in {name}')
    for name in ('queries_root', 'splits_dir', 'output_dir'):
        setattr(a, name, getattr(a, name).resolve())
    if not a.queries_root.is_dir() or not a.splits_dir.is_dir():
        p.error('Data and splits directories must exist')
    for source in (a.queries_root, a.splits_dir):
        if a.output_dir == source or source in a.output_dir.parents or a.output_dir in source.parents:
            p.error('Output must be outside and separate from data and splits trees')
    return a


def configure(kind, args):
    if kind == 'bag_full':
        from .bag_full_adapter import configure_bag
        return configure_bag(args)
    name = 'kmv' if kind == 'kmv' else ('vec_full_' if kind == 'vec_full' else 'full_') + args.scenario
    e = importlib.import_module('.' + name, __package__)
    e.EXTERNAL_ROOT, e.SPLIT_DIR, e.OUT_DIR = args.queries_root, args.splits_dir, args.output_dir
    e.TMP_DIR = e.OUT_DIR / 'tmp_python'
    e.SCORE_DIR = e.OUT_DIR / 'Scores'
    e.ANOMALY_CLASS = args.scenario.upper()
    e.FILE_TAG = '' if args.scenario == 's1' else '_s2'
    e.SPLIT_FILE_SUFFIX = 'shared_split_ids.csv' if args.scenario == 's1' else 's2_shared_split_ids.csv'
    e.ANOMALY_DIR_NAMES = ('{q}_anomalies', '{q}_annomalies') if args.scenario == 's1' else ('{q}_s2_anomalies', '{q}_s2_annomalies')
    e.SPLIT_POLICY = 'existing_manifest_exact_identity_no_resampling'
    e.EVALUATION_KIND = kind
    e.ALLOW_CREATE_SHARED_SPLITS = False
    e.RECREATE_SHARED_SPLITS = False
    e.K_NEIGHBORS = args.neighbors
    if kind == 'vec_full':
        e.PAIRWISE_N_JOBS = args.jobs
        e.PAIRWISE_WORKING_MEMORY_MB = args.working_memory_mb
        original_signature = e.current_job_signature
        def vector_signature(*positional, **keyword):
            original = original_signature(*positional, **keyword)
            return hashlib.sha256(json.dumps({'original': original, 'neighbors': args.neighbors,
                'jobs': args.jobs, 'working_memory_mb': args.working_memory_mb}, sort_keys=True).encode()).hexdigest()
        e.current_job_signature = vector_signature
        def vector_self_test():
            selected_k = e.K_NEIGHBORS
            try:
                e.K_NEIGHBORS = 1
                e.run_vectorization_self_test()
            finally:
                e.K_NEIGHBORS = selected_k
        e.warmup_numba = vector_self_test
    e.PARQUET_READ_ATTEMPTS, e.PARQUET_RETRY_DELAY_S = 3, 2.0
    e.MONITOR_DMESG = False
    e.clear_dmesg_best_effort = lambda: None
    e.check_dmesg_best_effort = lambda context: None
    e.split_manifest_path = lambda query, scale_name: e.SPLIT_DIR / f'{query.lower()}_{scale_name}_{e.SPLIT_FILE_SUFFIX}'
    # All three run_one implementations call one of these existing names.
    def existing(query, scale_name, normal_meta, anomaly_meta, train_k, test_normal_k, n_anom_test):
        e.validate_unique_identities(normal_meta, f'{query}/normal')
        e.validate_unique_identities(anomaly_meta, f'{query}/anomaly')
        scale = {'name': scale_name, 'train_k': train_k, 'test_normal_k': test_normal_k, 'n_anom_test': n_anom_test}
        manifest, path = read_manifest(e, query, scale)
        e.validate_split_manifest(manifest, normal_meta, anomaly_meta, train_k, test_normal_k, n_anom_test)
        return manifest, path
    e.load_or_create_split_manifest = existing
    e.load_existing_split_manifest = existing
    e.load_shared_split_manifest = existing
    e.load_required_split_manifest = existing
    if args.threads is not None and kind != 'vec_full':
        from numba import set_num_threads
        set_num_threads(args.threads)
    return e


def read_manifest(e, query, scale):
    path = e.split_manifest_path(query, scale['name'])
    if not path.is_file():
        raise FileNotFoundError(f'Existing split required: {path}; no split will be created')
    positions = {name: [] for name in e.SPLIT_NAMES}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            split = row['split']
            numeric, query_name = row['identity'].split('|', 1)
            if int(numeric) < 0 or not query_name or query_name.startswith('._'):
                raise ValueError(f'Invalid split identity: {row["identity"]}')
            if row.get('anomaly_class') and row['anomaly_class'].upper() != e.ANOMALY_CLASS:
                raise ValueError(f'Wrong anomaly scenario: {path}')
            if row.get('label') not in (None, '') and int(row['label']) != int(split == 'test_anom'):
                raise ValueError(f'Wrong label: {path}')
            opposite = 's2_' if e.ANOMALY_CLASS == 'S1' else 's1_'
            if split == 'test_anom' and (query_name.lower().startswith(opposite) or '_' + opposite in query_name.lower()):
                raise ValueError(f'Opposite anomaly scenario in {path}: {query_name}')
            positions[split].append(int(row['pos']))
    for split, pos in positions.items():
        if sorted(pos) != list(range(len(pos))):
            raise ValueError(f'Invalid/duplicate split positions: {path}/{split}')
    manifest = e.read_split_manifest(path, query, scale['name'])
    normal = [set(manifest[name]) for name in ('fit', 'calib', 'test_normal')]
    if normal[0] & normal[1] or normal[0] & normal[2] or normal[1] & normal[2]:
        raise ValueError(f'Overlapping normal partitions: {path}')
    if set(manifest['train_normal']) != normal[0] | normal[1]:
        raise ValueError(f'train_normal != fit union calib: {path}')
    calib = max(1, int(scale['train_k'] * e.CALIB_FRAC))
    sizes = [scale['train_k'], scale['train_k'] - calib, calib, scale['test_normal_k'], scale['n_anom_test']]
    for split, size in zip(e.SPLIT_NAMES, sizes):
        if len(manifest[split]) != size:
            raise ValueError(f'{path}/{split}: expected {size}, found {len(manifest[split])}')
    return manifest, path


def selected_scales(args):
    return [s for s in SCALES if s['name'] in args.scales]


def preflight(e, args, kind):
    snapshots = {}
    for qid in args.queries:
        query = f'q{qid}'
        for scale in selected_scales(args):
            _, path = read_manifest(e, query, scale)
            snapshots[str(path)] = sha(path)
        for pattern in (e.normal_glob(query), e.anomaly_glob(query)):
            files = e.parquet_files(pattern)
            if not files:
                raise FileNotFoundError(f'No Parquets: {pattern}')
            for file in files:
                path = Path(file)
                if any((path.parent / marker).exists() for marker in ('_full_run.json', '_run.json')) and not (path.parent / '_SUCCESS.json').exists():
                    raise ValueError(f'Incomplete materialisation: {path.parent}')
                stat = path.stat()
                snapshots[str(path)] = [stat.st_size, stat.st_mtime_ns]
                columns = e.pq.read_schema(path).names
                if not {'query_id', 'query_name'}.issubset(columns):
                    raise ValueError(f'Missing identity columns: {path}')
                for representation in args.representations:
                    method = methods_for(kind)[representation]
                    if kind != 'kmv':
                        if e.token_column_for_method(method) not in columns:
                            raise ValueError(f'Missing {representation} full column: {path}')
                    else:
                        for budget in args.kmv_k:
                            e.find_kmv_column(columns, method, budget, str(path))
        if args.check_splits:
            normal_meta = e.load_metadata(e.normal_glob(query))
            anomaly_meta = e.load_metadata(e.anomaly_glob(query))
            for scale in selected_scales(args):
                e.load_shared_split_manifest(query, scale['name'], normal_meta, anomaly_meta,
                    scale['train_k'], scale['test_normal_k'], scale['n_anom_test'])
                print(f'[CHECK OK] {query} {args.scenario} {scale["name"]}: existing identities/counts; no splits changed', flush=True)
    return snapshots


def result_csv(e, query, scale, method, budget):
    tag = e.FILE_TAG
    if e.EVALUATION_KIND == 'bag_full':
        return e.OUT_DIR / f'{query}{tag}_bag_full_{method}_{scale}.csv'
    if e.EVALUATION_KIND == 'vec_full':
        return e.OUT_DIR / f'{query}{tag}_vector_full_explicit_{method}_{scale}.csv'
    name = f'{query}{tag}_set_exact_twopass_{method}_{scale}.csv' if budget is None else f'{query}{tag}_set_kmv{budget}_exact_twopass_{method}_{scale}.csv'
    return e.OUT_DIR / name


def run_worker(e, args, kind):
    job = json.loads(args._worker_json.read_text())
    destination = Path(job['result_json'])
    query, method, budget = job['query'], job['method'], job['kmv_k']
    scale = next(s for s in SCALES if s['name'] == job['scale'])
    base = {'query': query, 'scale': scale['name'], 'method': method, 'anomaly_class': e.ANOMALY_CLASS,
            'kmv_k': budget, 'k_neighbors': args.neighbors, 'job_signature': job['signature']}
    try:
        manifest, split_path = read_manifest(e, query, scale)
        e.warmup_numba()
        if kind == 'bag_full':
            e.JOB_SIGNATURE = job['signature']
        kwargs = {'query': query, 'scale': scale, 'method': method}
        if kind == 'kmv':
            kwargs['kmv_k'] = budget
        row = e.run_one(**kwargs)
        if kind in ('vec_full', 'bag_full'):
            row['evaluation_signature'] = job['signature']
            # Preserve the source method/feature and score-file job signature.
        else:
            row.update(base)
        row['split_sha256'] = sha(split_path)
        row['split_members_and_order_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if kind not in ('vec_full', 'bag_full'):
            row['cost_protocol'] = f'parquet_load_plus_{"exact_full" if kind == "full" else "stored_kmv"}_set_{args.neighbors}nn'
            row['runtime_comparison_mode'] = f'{"full" if kind == "full" else "stored_kmv"}_ragged_set_{args.neighbors}nn'
        e.write_rows_csv(result_csv(e, query, scale['name'], method, budget), [row])
        dump(destination, row)
        dump(e.OUT_DIR / 'SplitIdentityAudit' / (job['stem'] + '.json'), {
            'manifest': str(split_path), 'split_sha256': row['split_sha256'],
            'identities': manifest, 'identity_matching': 'exact_numeric_id_and_query_name',
            'loading_order': 'manifest' if kind == 'bag_full' or (kind == 'full' and args.scenario == 's2') else 'parquet',
            'job_signature': job['signature']})
        return 0
    except Exception as exc:
        dump(destination, {**base, 'status': 'memory_error' if isinstance(exc, MemoryError) else 'error',
            'comparison_ready': False, 'error_type': type(exc).__name__, 'error': str(exc),
            'traceback': traceback.format_exc()})
        traceback.print_exc()
        return 1


def main(kind, argv=None):
    public = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(kind, public)
    if args.threads is not None and kind not in ('vec_full', 'bag_full'):
        os.environ['NUMBA_NUM_THREADS'] = str(args.threads)
    cache = args.output_dir / 'tmp_python'
    if not args.check_splits:
        cache.mkdir(parents=True, exist_ok=True)
        for key in ('TMPDIR', 'TEMP', 'TMP'):
            os.environ[key] = str(cache)
        os.environ['NUMBA_CACHE_DIR'] = str(cache / 'numba_cache')
    e = configure(kind, args)
    if args._worker_json:
        return run_worker(e, args, kind)
    inputs = preflight(e, args, kind)
    if args.check_splits:
        return 0
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                     if k not in ('rerun_completed', 'rerun_resource_failures', '_worker_json')}
    code = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    context = {'kind': kind, 'configuration': configuration, 'inputs': inputs, 'code': code}
    signature = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    for directory in ('Logs', 'WorkerResults', 'Scores', 'SplitIdentityAudit'):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    with (args.output_dir / '.evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_path = args.output_dir / 'evaluation_run.json'
        if run_path.exists() and json.loads(run_path.read_text()) != context:
            raise ValueError('Different configuration, input or code in output directory; use a new --output-dir')
        if not run_path.exists() and any(args.output_dir.rglob('*.csv')):
            raise ValueError('Existing results without a matching run manifest; use a new --output-dir')
        dump(run_path, context)
        results, errors = [], []
        if kind == 'bag_full':
            result_name, error_name = 'all_results.csv', 'all_errors.csv'
        elif kind == 'vec_full':
            prefix = 'full_vector_shared' if args.scenario == 's1' else 'full_vector_s2_shared'
            result_name, error_name = prefix + '_results.csv', prefix + '_errors.csv'
        elif kind == 'full':
            prefix = 'full_set_shared' if args.scenario == 's1' else 'full_set_s2_shared'
            result_name, error_name = prefix + '_results.csv', prefix + '_errors.csv'
        else:
            result_name = 'set_kmv_exact_all_fit_results_all.csv' if args.scenario == 's1' else 'set_kmv_s2_results_all.csv'
            error_name = 'set_kmv_exact_all_fit_errors.csv' if args.scenario == 's1' else 'set_kmv_s2_errors.csv'
        def publish():
            e.write_rows_csv(args.output_dir / result_name, results)
            e.write_rows_csv(args.output_dir / error_name, errors)
        budgets = args.kmv_k if kind == 'kmv' else [None]
        for qid in args.queries:
            query = f'q{qid}'
            for representation in args.representations:
                method = methods_for(kind)[representation]
                for budget in budgets:
                    stop = None
                    for scale in selected_scales(args):
                        stem = f'{query}_{args.scenario}_{scale["name"]}_{method}' + (f'_kmv{budget}' if budget else '')
                        destination = args.output_dir / 'WorkerResults' / (stem + '.json')
                        log = args.output_dir / 'Logs' / (stem + '.log')
                        cached = json.loads(destination.read_text()) if destination.exists() else {}
                        if cached and cached.get('evaluation_signature', cached.get('job_signature')) != signature:
                            raise ValueError(f'Checkpoint signature mismatch: {destination}')
                        artifacts_ready = kind not in ('vec_full', 'bag_full') or Path(cached.get('scores_file', '/nonexistent')).is_file()
                        if cached.get('status') == 'ok' and artifacts_ready and (not args.rerun_completed or stop):
                            results.append(cached)
                            publish()
                            print(f'[REUSE] {stem}', flush=True)
                            continue
                        if stop:
                            errors.append({'query': query, 'scale': scale['name'], 'method': method,
                                'kmv_k': budget, 'anomaly_class': e.ANOMALY_CLASS,
                                'status': 'skipped_after_resource_failure', 'stop_reason': stop,
                                'comparison_ready': False, 'job_signature': signature})
                            publish()
                            continue
                        resource_failure = cached.get('status') in ('timeout', 'memory_error', 'oom_or_sigkill')
                        if resource_failure and not args.rerun_resource_failures:
                            errors.append(cached)
                            stop = f'{scale["name"]}: {cached["status"]}'
                            publish()
                            continue
                        if destination.exists():
                            destination.unlink()
                        result_csv(e, query, scale['name'], method, budget).unlink(missing_ok=True)
                        audit_file = args.output_dir / 'SplitIdentityAudit' / (stem + '.json')
                        audit_file.unlink(missing_ok=True)
                        score_file = e.SCORE_DIR / (f'{query}{e.FILE_TAG}_{scale["name"]}_{method}_scores.npz' if kind in ('vec_full', 'bag_full') else f'{query}{e.FILE_TAG}_{scale["name"]}_{method}_kmv{budget}_scores.npz')
                        if kind in ('kmv', 'vec_full', 'bag_full'):
                            score_file.unlink(missing_ok=True)
                        job = {'query': query, 'scale': scale['name'], 'method': method,
                               'kmv_k': budget, 'result_json': str(destination), 'signature': signature, 'stem': stem}
                        task = args.output_dir / 'WorkerResults' / (stem + '.task.json')
                        dump(task, job)
                        command = [sys.executable, str(Path(sys.argv[0]).resolve()), *public, '--_worker-json', str(task)]
                        print(f'[START] {stem}', flush=True)
                        started = time.monotonic()
                        with log.open('w') as handle:
                            proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
                            try:
                                returncode = proc.wait(timeout=args.timeout_seconds)
                                timed_out = False
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                                returncode, timed_out = proc.returncode, True
                            except BaseException:
                                proc.kill()
                                proc.wait()
                                raise
                        if timed_out or not destination.exists():
                            row = {'query': query, 'scale': scale['name'], 'method': method,
                                   'kmv_k': budget, 'anomaly_class': e.ANOMALY_CLASS,
                                   'status': 'timeout' if timed_out else 'oom_or_sigkill' if returncode == -9 else 'worker_exit',
                                   'error': 'Worker timeout' if timed_out else 'Worker exited without result; inspect log/OS for cause',
                                   'returncode': returncode, 'comparison_ready': False, 'job_signature': signature}
                        else:
                            row = json.loads(destination.read_text())
                        row['log_file'] = str(log)
                        row['worker_wall_s'] = time.monotonic() - started
                        dump(destination, row)
                        if row['status'] == 'ok':
                            results.append(row)
                        else:
                            errors.append(row)
                            result_csv(e, query, scale['name'], method, budget).unlink(missing_ok=True)
                            audit_file.unlink(missing_ok=True)
                            if kind in ('kmv', 'vec_full', 'bag_full'):
                                score_file.unlink(missing_ok=True)
                            if row['status'] in ('timeout', 'memory_error', 'oom_or_sigkill'):
                                stop = f'{scale["name"]}: {row["status"]}'
                        publish()
                        print(f'[END {time.strftime("%Y-%m-%d %H:%M:%S %Z")}] {stem}: {row["status"]}', flush=True)
        dump(args.output_dir / 'evaluation_status.json', {'complete': not errors, 'results': len(results), 'errors': len(errors)})
        print(f'[DONE] results={len(results)} errors={len(errors)}', flush=True)
        return int(bool(errors))

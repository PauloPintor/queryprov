"""Command-line configuration for the existing CountSketch evaluation pipeline."""
import argparse
import fcntl
import json
import math
from pathlib import Path

from .cli import SCALES, dump, sha


def main(argv=None, bags=False):
    p = argparse.ArgumentParser(description=('Stored bag' if bags else 'Stored membership') + ' CountSketch -> L2 -> cosine 1-NN or 5-NN; no PSD')
    p.add_argument('--queries-root', type=Path, required=True)
    p.add_argument('--splits-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--scenario', choices=['s1', 's2'], required=True)
    p.add_argument('--queries', type=int, nargs='+', required=True)
    p.add_argument('--scales', nargs='+', choices=[s['name'] for s in SCALES], default=[s['name'] for s in SCALES])
    representations = ['token', 'witness'] if bags else ['lineage', 'atomic']
    p.add_argument('--representations', nargs='+', choices=representations, default=representations)
    p.add_argument('--dimensions', nargs='+', type=int, choices=[512, 1024], required=True)
    p.add_argument('--neighbors', type=int, choices=[1, 5], required=True)
    p.add_argument('--threads', type=int, default=4, help='BLAS threads')
    p.add_argument('--score-batch-rows', type=int, default=256)
    p.add_argument('--max-matrix-gib', type=float, default=4.0)
    p.add_argument('--timeout-seconds', type=float, default=18000)
    p.add_argument('--check-splits', action='store_true')
    a = p.parse_args(argv)
    if min(*a.queries, a.threads, a.score_batch_rows) <= 0 or any(not math.isfinite(v) or v <= 0 for v in (a.timeout_seconds, a.max_matrix_gib)):
        p.error('IDs, limits, threads and batch sizes must be positive')
    for name in ('queries', 'scales', 'representations', 'dimensions'):
        if len(set(getattr(a, name))) != len(getattr(a, name)):
            p.error(f'Duplicate {name}')
    for name in ('queries_root', 'splits_dir', 'output_dir'):
        setattr(a, name, getattr(a, name).resolve())
    if not a.queries_root.is_dir() or not a.splits_dir.is_dir():
        p.error('Existing data and split folders are required')
    for source in (a.queries_root, a.splits_dir):
        if a.output_dir == source or a.output_dir in source.parents or source in a.output_dir.parents:
            p.error('Output must be outside both input trees')
    if bags:
        from . import bag_countsketch as e
    else:
        from . import vec_countsketch as e
    e.INPUT_ROOT, e.OUT_DIR = a.queries_root, a.output_dir
    e.SPLIT_DIRS = {a.scenario.upper(): a.splits_dir}
    e.ANOMALY_CLASSES = (a.scenario.upper(),)
    e.QUERY_IDS = tuple(a.queries)
    names = {'token': 'bag_lineage', 'witness': 'bag_atomic_why'} if bags else {'lineage': 'lineage', 'atomic': 'atomic_why'}
    e.FEATURES = tuple(names[name] for name in a.representations)
    e.SKETCH_DIMS = tuple(a.dimensions)
    e.SCALES = tuple(s for s in SCALES if s['name'] in a.scales)
    e.K_NEIGHBORS = a.neighbors
    e.BLAS_THREADS = a.threads
    e.SCORE_BATCH_ROWS = a.score_batch_rows
    e.MAX_MATRIX_GIB = a.max_matrix_gib
    e.WORKER_TIMEOUT_SECONDS = a.timeout_seconds
    if a.check_splits:
        jobs = e.build_jobs()
        print(f'[CHECK OK] {len(jobs)} configurations; splits and input identities verified; no scoring')
        return 0
    context = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k != 'check_splits'}
    context['family'] = 'bags' if bags else 'membership'
    context['code'] = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.output_dir / '.cli.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = a.output_dir / 'evaluation_cli.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != context:
                raise ValueError('Output has a different configuration/code; use a new output directory')
        elif any(a.output_dir.rglob('*.csv')) or any(a.output_dir.rglob('*.npz')):
            raise ValueError('Existing unversioned outputs; use a new output directory')
        dump(manifest, context)
        return e.main()

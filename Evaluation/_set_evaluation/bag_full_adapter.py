"""Connect the bag scientific engines to the existing full/set scheduler."""
import glob
import importlib
from sklearn import set_config


def configure_bag(args):
    from .cli import SCALES, read_manifest
    e = importlib.import_module('.bag_full_' + args.scenario, __package__)
    metadata = importlib.import_module('.vec_full_' + args.scenario, __package__)
    e.BASE_DIR = args.queries_root
    e.SPLIT_DIR, e.OUT_DIR = args.splits_dir, args.output_dir
    e.SCORE_DIR = e.OUT_DIR / 'Scores'
    e.FILE_TAG = '' if args.scenario == 's1' else '_s2'
    e.ANOMALY_CLASS = args.scenario.upper()
    e.EVALUATION_KIND = 'bag_full'
    e.KNN_K, e.N_JOBS = args.neighbors, args.jobs
    e.SCALES = SCALES
    set_config(working_memory=args.working_memory_mb)
    suffix = 'shared_split_ids.csv' if args.scenario == 's1' else 's2_shared_split_ids.csv'
    e.split_manifest_path = lambda query, scale_name: e.SPLIT_DIR / f'q{str(query).lower().removeprefix("q")}_{scale_name}_{suffix}'
    e.SPLIT_NAMES = ('train_normal', 'fit', 'calib', 'test_normal', 'test_anom')
    e.read_split_manifest = metadata.read_split_manifest
    e.validate_unique_identities = metadata.validate_unique_identities
    e.validate_split_manifest = metadata.validate_split_manifest
    e.load_metadata = metadata.load_metadata
    e.normal_glob = lambda query: e.resolve_query_patterns(int(query[1:]))[0]
    e.anomaly_glob = lambda query: e.resolve_query_patterns(int(query[1:]))[1]
    e.parquet_files = lambda pattern: sorted(p for p in glob.glob(pattern, recursive=True) if e.is_source_parquet_path(p))
    e.token_column_for_method = lambda method: e.bag_column_for_feature(method)[0]
    e.warmup_numba = lambda: None
    def existing(query, scale_name, normal_meta, anomaly_meta, train_k, test_normal_k, n_anom_test):
        manifest, path = read_manifest(e, query, next(s for s in SCALES if s['name'] == scale_name))
        e.validate_unique_identities(normal_meta, f'{query}/normal')
        e.validate_unique_identities(anomaly_meta, f'{query}/anomaly')
        e.validate_split_manifest(manifest, normal_meta, anomaly_meta, train_k, test_normal_k, n_anom_test)
        return manifest, path
    e.load_shared_split_manifest = existing
    e.read_bag_manifest = lambda query_id, scale_name: read_manifest(e, f'q{query_id}', next(s for s in SCALES if s['name'] == scale_name))
    e.run_one = lambda query, scale, method: e.run_worker(int(query[1:]), scale['name'], method, 'knn_distance', 'cosine')
    return e

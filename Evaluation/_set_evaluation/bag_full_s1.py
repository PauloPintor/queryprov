# Bag-only scientific engine extracted from Vec_Full_Bag.py.
from __future__ import annotations
import argparse
import csv
import gc
import glob
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
FEATURES = ['token_bag', 'witness_bag']
UNSUP_METHODS = ['knn_distance']
DISTANCE_METRICS = ['cosine']
RANDOM_SEED = 13
CALIB_FRAC = 0.2
THRESHOLD_PERCENTILES = [95, 99]
BOOTSTRAP_B = 1000
PARQUET_BATCH_SIZE = 4096
DERIVED_PARQUET_SUFFIXES = ('.bagcompact.parquet', '.bagtfcompact.parquet')
MAX_FILE_WARNINGS = 10
METADATA_COLUMNS = ['query_id', 'template_id', 'query_name', 'subtype']

@dataclass
class FullDataset:
    feature: str
    query_ids: np.ndarray
    template_ids: np.ndarray
    query_names: np.ndarray
    subtypes: np.ndarray
    labels: np.ndarray
    tokens: np.ndarray
    values: np.ndarray
    offsets: np.ndarray

    def __len__(self) -> int:
        return int(len(self.labels))

@dataclass
class FullDatasetBuilder:
    feature: str
    query_ids: List[int] = field(default_factory=list)
    template_ids: List[int] = field(default_factory=list)
    query_names: List[str] = field(default_factory=list)
    subtypes: List[str] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    token_chunks: List[np.ndarray] = field(default_factory=list)
    value_chunks: List[np.ndarray] = field(default_factory=list)
    offsets: List[int] = field(default_factory=lambda: [0])

    def _append_meta(self, query_id: int, template_id: int, query_name: str, subtype: str, label: int) -> None:
        self.query_ids.append(int(query_id))
        self.template_ids.append(int(template_id))
        self.query_names.append(str(query_name))
        self.subtypes.append(str(subtype))
        self.labels.append(int(label))

    def append_bag(self, query_id: int, template_id: int, query_name: str, subtype: str, label: int, tokens: np.ndarray, values: np.ndarray) -> None:
        self._append_meta(query_id=query_id, template_id=template_id, query_name=query_name, subtype=subtype, label=label)
        tokens = np.asarray(tokens, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)
        if len(tokens) != len(values):
            raise ValueError('tokens e values têm tamanhos diferentes')
        if len(tokens) > 0:
            self.token_chunks.append(tokens)
            self.value_chunks.append(values)
        self.offsets.append(self.offsets[-1] + len(tokens))

    def finalize(self) -> FullDataset:
        tokens = np.concatenate(self.token_chunks).astype(np.int64, copy=False) if self.token_chunks else np.empty(0, dtype=np.int64)
        values = np.concatenate(self.value_chunks).astype(np.float32, copy=False) if self.value_chunks else np.empty(0, dtype=np.float32)
        return FullDataset(feature=self.feature, query_ids=np.asarray(self.query_ids, dtype=np.int64), template_ids=np.asarray(self.template_ids, dtype=np.int32), query_names=np.asarray(self.query_names, dtype=object), subtypes=np.asarray(self.subtypes, dtype=object), labels=np.asarray(self.labels, dtype=np.int8), tokens=tokens, values=values, offsets=np.asarray(self.offsets, dtype=np.int64))

@dataclass
class LoadReport:
    pattern: str
    feature: str
    requested_items: Optional[int]
    files_found: int = 0
    files_read: int = 0
    files_missing_feature: int = 0
    files_with_error: int = 0
    rows_seen: int = 0
    rows_loaded: int = 0
    rows_invalid: int = 0
    tokens_loaded: int = 0
    missing_columns: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

@dataclass
class SparseTransform:
    feature: str
    vocab: Dict[int, int]

    @property
    def dim(self) -> int:
        return len(self.vocab)
DatasetSlice = Tuple[FullDataset, np.ndarray]

def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys.platform != 'darwin' else value / (1024.0 * 1024.0)

def row_identity(data: FullDataset, i: int) -> str:
    return f'{int(data.query_ids[i])}|{str(data.query_names[i])}'

def identity_from_row(row: Dict[str, Any]) -> str:
    query_id = _safe_int(row.get('query_id'), -1)
    query_name = str(row.get('query_name', ''))
    return f'{query_id}|{query_name}'

def is_source_parquet_path(path: str | Path) -> bool:
    name = Path(path).name
    lower_name = name.lower()
    return lower_name.endswith('.parquet') and (not name.startswith('._')) and (not lower_name.endswith(DERIVED_PARQUET_SUFFIXES))

def split_manifest_path(query_id: int, scale_name: str) -> Path:
    return SPLIT_DIR / f'q{query_id}_{scale_name}_shared_split_ids.csv'

def read_bag_manifest(query_id: int, scale_name: str) -> Tuple[Dict[str, List[str]], Path]:
    path = split_manifest_path(query_id, scale_name)
    if not path.exists():
        raise FileNotFoundError(f'Manifesto de splits não encontrado: {path}. Executa primeiro o programa Full Lineage/Atomic.')
    split_names = ('train_normal', 'fit', 'calib', 'test_normal', 'test_anom')
    positioned: Dict[str, List[Tuple[int, str]]] = {name: [] for name in split_names}
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'query', 'scale', 'split', 'pos', 'identity'}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: faltam colunas {sorted(missing)}')
        expected_query = f'q{query_id}'
        for row in reader:
            if str(row['query']).lower() != expected_query:
                raise ValueError(f'{path}: contém uma query diferente.')
            if str(row['scale']) != scale_name:
                raise ValueError(f'{path}: contém uma escala diferente.')
            split_name = str(row['split'])
            if split_name not in positioned:
                raise ValueError(f'{path}: split desconhecido {split_name!r}')
            positioned[split_name].append((int(row['pos']), str(row['identity'])))
    manifest: Dict[str, List[str]] = {}
    for split_name, rows in positioned.items():
        rows.sort(key=lambda item: item[0])
        identities = [identity for _, identity in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f'{path}: identidades repetidas em {split_name}')
        manifest[split_name] = identities
    if set(manifest['train_normal']) != set(manifest['fit']) | set(manifest['calib']):
        raise ValueError(f'{path}: train_normal não corresponde a fit + calib.')
    if set(manifest['fit']) & set(manifest['calib']):
        raise ValueError(f'{path}: fit e calib têm execuções em comum.')
    if (set(manifest['fit']) | set(manifest['calib'])) & set(manifest['test_normal']):
        raise ValueError(f'{path}: treino e teste normal têm execuções em comum.')
    return (manifest, path)

def _safe_int(value: Any, default: int=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default

def _safe_float(value: Any, default: float=0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default

def ratio(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) > 0.0 else 0.0

def as_py_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, 'as_py'):
        value = value.as_py()
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]

def extract_bag_arrays(value: Any, token_key: str, count_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    token_bag:
        token_tf: list<struct<token, tf>>

    witness_bag:
        witness_tf: list<struct<witness_hash, tf>>

    Mantém contagens/frequências.
    """
    acc: Dict[int, float] = defaultdict(float)
    for entry in as_py_list(value):
        if entry is None:
            continue
        if hasattr(entry, 'as_py'):
            entry = entry.as_py()
        if not isinstance(entry, dict):
            continue
        token = entry.get(token_key)
        count = entry.get(count_key, 1.0)
        if token is None:
            continue
        try:
            tok_i = int(token)
            val_f = float(count)
        except (TypeError, ValueError, OverflowError):
            continue
        if val_f == 0.0:
            continue
        acc[tok_i] += val_f
    if not acc:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
    toks = np.asarray(sorted(acc.keys()), dtype=np.int64)
    vals = np.asarray([acc[int(t)] for t in toks], dtype=np.float32)
    return (toks, vals)

def bag_column_for_feature(feature: str) -> Tuple[str, str, str]:
    if feature == 'token_bag':
        return ('token_tf', 'token', 'tf')
    if feature == 'witness_bag':
        return ('witness_tf', 'witness_hash', 'tf')
    raise ValueError(feature)

def required_columns(feature: str) -> List[str]:
    if feature in {'token_bag', 'witness_bag'}:
        col, _, _ = bag_column_for_feature(feature)
        return METADATA_COLUMNS + [col]
    raise ValueError(feature)

def resolve_query_patterns(query_id: int) -> Tuple[str, str]:
    q = f'q{query_id}'
    q_upper = f'Q{query_id}'
    normal_candidates = [BASE_DIR / q_upper / f'{q}_normal', BASE_DIR / f'{q}_normal']
    anomaly_candidates = [BASE_DIR / q_upper / f'{q}_anomalies', BASE_DIR / q_upper / f'{q}_annomalies', BASE_DIR / f'{q}_anomalies', BASE_DIR / f'{q}_annomalies']
    normal = next((p for p in normal_candidates if p.exists()), None)
    if normal is None:
        raise FileNotFoundError('Normais não encontrados. Candidatos:\n' + '\n'.join((str(p) for p in normal_candidates)))
    anomaly = next((p for p in anomaly_candidates if p.exists()), None)
    if anomaly is None:
        raise FileNotFoundError('Anomalias não encontradas. Candidatos:\n' + '\n'.join((str(p) for p in anomaly_candidates)))
    return (str(normal / '**' / '*.parquet'), str(anomaly / '**' / '*.parquet'))

def load_full_dataset(pattern: str, label: int, feature: str, selected_identities: set[str], default_template_id: int) -> Tuple[FullDataset, LoadReport]:
    files = sorted((fp for fp in glob.glob(pattern, recursive=True) if is_source_parquet_path(fp)))
    report = LoadReport(pattern=pattern, feature=feature, requested_items=len(selected_identities), files_found=len(files))
    if not files:
        raise FileNotFoundError(f'Nenhum Parquet encontrado: {pattern}')
    needed = required_columns(feature)
    builder = FullDatasetBuilder(feature=feature)
    warnings = 0
    t0 = time.perf_counter()
    bag_col = ''
    token_key = ''
    count_key = ''
    target = len(selected_identities)
    if feature in {'token_bag', 'witness_bag'}:
        bag_col, token_key, count_key = bag_column_for_feature(feature)
    for path in files:
        if report.rows_loaded >= target:
            break
        try:
            pf = pq.ParquetFile(path)
            schema_names = pf.schema_arrow.names
        except Exception as exc:
            report.files_with_error += 1
            if warnings < MAX_FILE_WARNINGS:
                print(f'[WARN] Falha a abrir {path}: {exc}', flush=True)
                warnings += 1
            continue
        missing = [c for c in needed if c not in schema_names]
        if missing:
            report.files_missing_feature += 1
            for c in missing:
                report.missing_columns[c] += 1
            if warnings < MAX_FILE_WARNINGS:
                print(f'[WARN] Ignorado {path}: faltam {missing}', flush=True)
                warnings += 1
            continue
        try:
            for batch in pf.iter_batches(columns=needed, batch_size=PARQUET_BATCH_SIZE, use_threads=True):
                rows = batch.to_pylist()
                for row in rows:
                    if report.rows_loaded >= target:
                        break
                    report.rows_seen += 1
                    try:
                        identity = identity_from_row(row)
                        if identity not in selected_identities:
                            continue
                        query_id = _safe_int(row.get('query_id'), -1)
                        template_id = _safe_int(row.get('template_id'), default_template_id)
                        query_name = str(row.get('query_name', ''))
                        subtype = str(row.get('subtype', ''))
                        if feature in {'token_bag', 'witness_bag'}:
                            tokens, values = extract_bag_arrays(row.get(bag_col), token_key=token_key, count_key=count_key)
                            builder.append_bag(query_id=query_id, template_id=template_id, query_name=query_name, subtype=subtype, label=label, tokens=tokens, values=values)
                            report.tokens_loaded += int(len(tokens))
                        else:
                            raise ValueError(feature)
                        report.rows_loaded += 1
                    except Exception:
                        report.rows_invalid += 1
                del rows
                del batch
                if report.rows_loaded >= target:
                    break
            report.files_read += 1
        except Exception as exc:
            report.files_with_error += 1
            if warnings < MAX_FILE_WARNINGS:
                print(f'[WARN] Falha a ler {path}: {exc}', flush=True)
                warnings += 1
        gc.collect()
    data = builder.finalize()
    loaded_identities = [row_identity(data, i) for i in range(len(data))]
    if len(loaded_identities) != len(set(loaded_identities)):
        raise ValueError(f'{feature}: foram carregadas identidades duplicadas.')
    missing_identities = selected_identities.difference(loaded_identities)
    if missing_identities:
        raise ValueError(f'{feature}: faltam {len(missing_identities):,} execuções do manifesto; exemplos={sorted(missing_identities)[:5]}')
    print(f'[LOAD DONE] feature={feature} label={label} files={report.files_read}/{report.files_found} rows={report.rows_loaded:,} tokens={report.tokens_loaded:,} time={time.perf_counter() - t0:.2f}s peak={peak_rss_mb():.1f}MB', flush=True)
    return (data, report)

def indices_for_identities(data: FullDataset, identities: List[str], split_name: str) -> np.ndarray:
    index_by_identity: Dict[str, int] = {}
    for i in range(len(data)):
        identity = row_identity(data, i)
        if identity in index_by_identity:
            raise ValueError(f'Identidade duplicada no dataset: {identity}')
        index_by_identity[identity] = i
    missing = [identity for identity in identities if identity not in index_by_identity]
    if missing:
        raise ValueError(f'Split {split_name}: {len(missing):,} identidades ausentes; exemplos={missing[:5]}')
    return np.asarray([index_by_identity[identity] for identity in identities], dtype=np.int64)

def labels_from_slices(parts: Sequence[DatasetSlice]) -> np.ndarray:
    labels: List[np.ndarray] = []
    for data, idx in parts:
        labels.append(data.labels[idx].astype(np.int32, copy=False))
    if not labels:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(labels).astype(np.int32, copy=False)

def build_vocab_from_sparse_slices(parts: Sequence[DatasetSlice]) -> Dict[int, int]:
    vocab: Dict[int, int] = {}
    for data, idxs in parts:
        for raw_i in idxs:
            i = int(raw_i)
            start = int(data.offsets[i])
            end = int(data.offsets[i + 1])
            for p in range(start, end):
                tok = int(data.tokens[p])
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab

def build_csr_from_sparse_slices(parts: Sequence[DatasetSlice], vocab: Dict[int, int], binary: bool) -> csr_matrix:
    n_rows = sum((len(idxs) for _, idxs in parts))
    n_cols = len(vocab)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    indices: List[int] = []
    data_values: List[float] = []
    row_out = 0
    for data, idxs in parts:
        for raw_i in idxs:
            i = int(raw_i)
            start = int(data.offsets[i])
            end = int(data.offsets[i + 1])
            row_acc: Dict[int, float] = defaultdict(float)
            for p in range(start, end):
                col = vocab.get(int(data.tokens[p]))
                if col is None:
                    raise ValueError('Elemento selecionado ausente do vocabulário comum')
                if binary:
                    row_acc[col] = 1.0
                else:
                    row_acc[col] += float(data.values[p])
            if row_acc:
                cols = sorted(row_acc)
                indices.extend(cols)
                data_values.extend([row_acc[c] for c in cols])
            row_out += 1
            indptr[row_out] = len(indices)
    return csr_matrix((np.asarray(data_values, dtype=np.float32), np.asarray(indices, dtype=np.int32), indptr), shape=(n_rows, n_cols), dtype=np.float32)

def fit_sparse_feature(feature: str, fit_parts: Sequence[DatasetSlice], vocabulary_parts: Sequence[DatasetSlice]) -> Tuple[csr_matrix, SparseTransform]:
    vocab = build_vocab_from_sparse_slices(vocabulary_parts)
    if not vocab:
        raise ValueError(f'Vocabulário vazio para {feature}')
    binary = False
    Xfit = build_csr_from_sparse_slices(parts=fit_parts, vocab=vocab, binary=binary).tocsr()
    transform = SparseTransform(feature=feature, vocab=vocab)
    return (Xfit, transform)

def transform_sparse_feature(parts: Sequence[DatasetSlice], transform: SparseTransform) -> csr_matrix:
    binary = False
    return build_csr_from_sparse_slices(parts=parts, vocab=transform.vocab, binary=binary).tocsr()

def fit_feature(feature: str, fit_parts: Sequence[DatasetSlice], vocabulary_parts: Sequence[DatasetSlice]) -> Tuple[Any, Any]:
    if feature in {'token_bag', 'witness_bag'}:
        return fit_sparse_feature(feature, fit_parts, vocabulary_parts)
    raise ValueError(feature)

def transform_feature(feature: str, parts: Sequence[DatasetSlice], transform: Any) -> Any:
    if feature in {'token_bag', 'witness_bag'}:
        return transform_sparse_feature(parts, transform)
    raise ValueError(feature)

def transform_dim(transform: Any) -> int:
    if isinstance(transform, SparseTransform):
        return int(transform.dim)
    return -1

def representation_name(feature: str) -> str:
    return {'token_bag': 'full_sparse_token_bag_counts', 'witness_bag': 'full_sparse_witness_bag_counts'}[feature]

def fit_knn(Xfit: Any, metric: str) -> NearestNeighbors:
    k = min(KNN_K, max(1, Xfit.shape[0]))
    model = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='brute', n_jobs=N_JOBS)
    model.fit(Xfit)
    return model

def score_knn(model: NearestNeighbors, X: Any) -> np.ndarray:
    dists, _ = model.kneighbors(X, return_distance=True)
    return np.asarray(dists[:, 0] if KNN_K == 1 else np.mean(dists, axis=1, dtype=np.float64), dtype=np.float32)

def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float('nan')

def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float('nan')
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float('nan')

def compute_threshold_bootstrap(calib_scores: np.ndarray, percentile: int, B: int, seed: int) -> Tuple[float, float, float]:
    if len(calib_scores) == 0:
        return (float('nan'), float('nan'), float('nan'))
    if B <= 0:
        thr = float(np.percentile(calib_scores, percentile))
        return (thr, float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    thresholds = np.zeros(B, dtype=np.float64)
    for b in range(B):
        sample = rng.choice(calib_scores, size=len(calib_scores), replace=True)
        thresholds[b] = np.percentile(sample, percentile)
    return (float(np.mean(thresholds)), float(np.percentile(thresholds, 2.5)), float(np.percentile(thresholds, 97.5)))

def evaluate_scores(y_true: np.ndarray, calib_scores: np.ndarray, test_scores: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {'ap': safe_ap(y_true, test_scores), 'auc': safe_auc(y_true, test_scores), 'calib_score_min': float(np.min(calib_scores)), 'calib_score_median': float(np.median(calib_scores)), 'calib_score_max': float(np.max(calib_scores)), 'test_score_min': float(np.min(test_scores)), 'test_score_median': float(np.median(test_scores)), 'test_score_max': float(np.max(test_scores))}
    for pct in THRESHOLD_PERCENTILES:
        thr, ci_lo, ci_hi = compute_threshold_bootstrap(calib_scores=calib_scores, percentile=pct, B=BOOTSTRAP_B, seed=RANDOM_SEED)
        pred = (test_scores > thr).astype(np.int32)
        suffix = f'_pct{pct}'
        out[f'threshold{suffix}'] = float(thr)
        out[f'thr_ci_lo{suffix}'] = float(ci_lo)
        out[f'thr_ci_hi{suffix}'] = float(ci_hi)
        out[f'f1{suffix}'] = float(f1_score(y_true, pred, zero_division=0))
        out[f'precision{suffix}'] = float(precision_score(y_true, pred, zero_division=0))
        out[f'recall{suffix}'] = float(recall_score(y_true, pred, zero_division=0))
        out[f'n_pred{suffix}'] = int(pred.sum())
    return out

def matrix_nbytes(X: Any) -> int:
    if issparse(X):
        X = X.tocsr()
        return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
    return int(getattr(X, 'nbytes', 0))

def matrix_info(X: Any) -> Dict[str, Any]:
    return {'x_shape': str(X.shape), 'x_nnz': int(X.nnz) if issparse(X) else '', 'x_kb': float(matrix_nbytes(X) / 1024.0), 'dtype': str(X.dtype), 'is_sparse': bool(issparse(X))}

def scale_by_name(name: str) -> Dict[str, Any]:
    for scale in SCALES:
        if scale['name'] == name:
            return scale
    raise KeyError(name)

def run_worker(query_id: int, scale_name: str, feature: str, method: str, metric: str) -> Dict[str, Any]:
    if feature not in FEATURES:
        raise ValueError(feature)
    if method not in UNSUP_METHODS:
        raise ValueError(method)
    if metric not in DISTANCE_METRICS:
        raise ValueError(metric)
    scale = scale_by_name(scale_name)
    normal_glob, anomaly_glob = resolve_query_patterns(query_id)
    manifest, manifest_path = read_bag_manifest(query_id=query_id, scale_name=scale_name)
    fit_ids = manifest['fit']
    calib_ids = manifest['calib']
    test_normal_ids = manifest['test_normal']
    test_anom_ids = manifest['test_anom']
    if len(manifest['train_normal']) != int(scale['train_k']):
        raise ValueError(f"{manifest_path}: train_normal={len(manifest['train_normal']):,}, esperado={int(scale['train_k']):,}")
    if len(test_normal_ids) != int(scale['test_normal_k']):
        raise ValueError(f"{manifest_path}: test_normal={len(test_normal_ids):,}, esperado={int(scale['test_normal_k']):,}")
    if len(test_anom_ids) != int(scale['n_anom_test']):
        raise ValueError(f"{manifest_path}: test_anom={len(test_anom_ids):,}, esperado={int(scale['n_anom_test']):,}")
    total_start = time.perf_counter()
    print(f'[PATHS] query=q{query_id} normal={normal_glob} anomaly={anomaly_glob}', flush=True)
    load_start = time.perf_counter()
    normals, normal_report = load_full_dataset(pattern=normal_glob, label=0, feature=feature, selected_identities=set(fit_ids + calib_ids + test_normal_ids), default_template_id=query_id)
    anomalies, anomaly_report = load_full_dataset(pattern=anomaly_glob, label=1, feature=feature, selected_identities=set(test_anom_ids), default_template_id=query_id)
    load_s = time.perf_counter() - load_start
    split_start = time.perf_counter()
    fit_normal_idx = indices_for_identities(normals, fit_ids, 'fit')
    calib_normal_idx = indices_for_identities(normals, calib_ids, 'calib')
    test_normal_idx = indices_for_identities(normals, test_normal_ids, 'test_normal')
    test_anom_idx = indices_for_identities(anomalies, test_anom_ids, 'test_anom')
    fit_parts: List[DatasetSlice] = [(normals, fit_normal_idx)]
    calib_parts: List[DatasetSlice] = [(normals, calib_normal_idx)]
    test_normal_parts: List[DatasetSlice] = [(normals, test_normal_idx)]
    test_anom_parts: List[DatasetSlice] = [(anomalies, test_anom_idx)]
    y_true = np.concatenate([np.zeros(len(test_normal_idx), dtype=np.int8), np.ones(len(test_anom_idx), dtype=np.int8)])
    split_s = time.perf_counter() - split_start
    contamination = float(y_true.sum() / max(1, len(y_true)))
    print(f'[SPLIT] fit={len(fit_normal_idx):,} calib={len(calib_normal_idx):,} test_normal={len(test_normal_idx):,} test_anom={len(test_anom_idx):,} contamination={contamination:.4f}', flush=True)
    build_start = time.perf_counter()
    Xfit, transform = fit_feature(feature=feature, fit_parts=fit_parts, vocabulary_parts=fit_parts + calib_parts + test_normal_parts + test_anom_parts)
    Xcalib = transform_feature(feature=feature, parts=calib_parts, transform=transform)
    Xtest_normal = transform_feature(feature=feature, parts=test_normal_parts, transform=transform)
    Xtest_anom = transform_feature(feature=feature, parts=test_anom_parts, transform=transform)
    build_s = time.perf_counter() - build_start
    dim = transform_dim(transform)
    xfit_info = matrix_info(Xfit)
    xcalib_info = matrix_info(Xcalib)
    xtest_normal_info = matrix_info(Xtest_normal)
    xtest_anom_info = matrix_info(Xtest_anom)
    print(f"[MATRIX] feature={feature} dim={dim:,} metric={metric} Xfit={Xfit.shape} nnz_fit={getattr(Xfit, 'nnz', '')} Xcalib={Xcalib.shape} nnz_calib={getattr(Xcalib, 'nnz', '')} Xtest_normal={Xtest_normal.shape} Xtest_anom={Xtest_anom.shape} build={build_s:.2f}s peak={peak_rss_mb():.1f}MB", flush=True)
    if method == 'knn_distance':
        fit_start = time.perf_counter()
        model = fit_knn(Xfit, metric=metric)
        fit_s = time.perf_counter() - fit_start
        score_calib_start = time.perf_counter()
        calib_scores = score_knn(model, Xcalib)
        score_calib_s = time.perf_counter() - score_calib_start
        score_normal_start = time.perf_counter()
        normal_scores = score_knn(model, Xtest_normal)
        score_normal_s = time.perf_counter() - score_normal_start
        score_anom_start = time.perf_counter()
        anom_scores = score_knn(model, Xtest_anom)
        score_anom_s = time.perf_counter() - score_anom_start
    else:
        raise ValueError(method)
    score_total_s = score_calib_s + score_normal_s + score_anom_s
    model_score_s = fit_s + score_total_s
    test_scores = np.concatenate([normal_scores, anom_scores]).astype(np.float32, copy=False)
    metrics_start = time.perf_counter()
    result = evaluate_scores(y_true=y_true, calib_scores=calib_scores, test_scores=test_scores)
    metrics_s = time.perf_counter() - metrics_start
    preprocess_s = load_s + split_s + build_s
    total_s = preprocess_s + fit_s + score_total_s
    representation_total_kb = float(xfit_info['x_kb']) + float(xcalib_info['x_kb']) + float(xtest_normal_info['x_kb']) + float(xtest_anom_info['x_kb'])
    result.update({'query': f'q{query_id}', 'query_id': int(query_id), 'scale': scale_name, 'feature': feature, 'method': method, 'metric': metric, 'representation': representation_name(feature), 'representation_family': 'full_vector', 'representation_dim': int(dim), 'uses_kmv': False, 'uses_hashing_vectorizer': False, 'uses_svd': False, 'uses_tfidf': False, 'uses_reduction': False, 'train_k': int(scale['train_k']), 'test_normal_k': int(scale['test_normal_k']), 'n_anom_test_config': int(scale['n_anom_test']), 'n_train': int(len(manifest['train_normal'])), 'n_fit': int(len(fit_normal_idx)), 'n_fit_actual': int(len(fit_normal_idx)), 'n_calib': int(len(calib_normal_idx)), 'n_test': int(len(y_true)), 'n_test_normal': int(len(test_normal_idx)), 'n_test_anom': int(y_true.sum()), 'contamination': float(contamination), 'calib_frac': float(CALIB_FRAC), 'feature_dim': int(dim), 'xfit_shape': xfit_info['x_shape'], 'xfit_nnz': xfit_info['x_nnz'], 'xfit_kb': xfit_info['x_kb'], 'xfit_dtype': xfit_info['dtype'], 'xcalib_shape': xcalib_info['x_shape'], 'xcalib_nnz': xcalib_info['x_nnz'], 'xcalib_kb': xcalib_info['x_kb'], 'xcalib_dtype': xcalib_info['dtype'], 'xtest_normal_shape': xtest_normal_info['x_shape'], 'xtest_normal_nnz': xtest_normal_info['x_nnz'], 'xtest_normal_kb': xtest_normal_info['x_kb'], 'xtest_normal_dtype': xtest_normal_info['dtype'], 'xtest_anom_shape': xtest_anom_info['x_shape'], 'xtest_anom_nnz': xtest_anom_info['x_nnz'], 'xtest_anom_kb': xtest_anom_info['x_kb'], 'xtest_anom_dtype': xtest_anom_info['dtype'], 'xtest_shape': str((len(y_true), dim)), 'xtest_nnz': int(xtest_normal_info['x_nnz']) + int(xtest_anom_info['x_nnz']) if xtest_normal_info['x_nnz'] != '' else '', 'xtest_kb': float(xtest_normal_info['x_kb'] + xtest_anom_info['x_kb']), 'representation_total_kb': representation_total_kb, 'matrix_total_kb': representation_total_kb, 'knn_k': int(KNN_K), 'threshold_percentiles': ','.join(map(str, THRESHOLD_PERCENTILES)), 'bootstrap_b': int(BOOTSTRAP_B), 'load_s': float(load_s), 'split_s': float(split_s), 'build_s': float(build_s), 'representation_build_s': float(build_s), 'fit_s': float(fit_s), 'score_calib_s': float(score_calib_s), 'score_normal_s': float(score_normal_s), 'score_anom_s': float(score_anom_s), 'score_s': float(score_total_s), 'score_total_s': float(score_total_s), 'model_score_s': float(model_score_s), 'metrics_s': float(metrics_s), 'preprocess_s': float(preprocess_s), 'preprocess_total_s': float(preprocess_s), 'total_s': float(total_s), 'end_to_end_s': float(time.perf_counter() - total_start), 'runtime_s': float(total_s), 'peak_rss_mb': float(peak_rss_mb()), 'split_manifest': str(manifest_path), 'uses_shared_split': True, 'shared_split_dir': str(SPLIT_DIR), 'shared_split_file': str(manifest_path), 'load_protocol': 'selected_full_rows_then_all_selected_splits_feature_mapping', 'vocabulary_scope': 'union_of_all_selected_splits_labels_ignored', 'quality_protocol': 'shared_split_threshold_from_calib_percentile', 'cost_protocol': 'parquet_load_plus_full_vector_build_plus_detector', 'runtime_comparison_mode': 'full_vector_worker_isolated', 'comparison_ready': True, 'normal_files_found': int(normal_report.files_found), 'normal_files_read': int(normal_report.files_read), 'normal_files_missing_feature': int(normal_report.files_missing_feature), 'normal_files_with_error': int(normal_report.files_with_error), 'normal_rows_seen': int(normal_report.rows_seen), 'normal_rows_loaded': int(normal_report.rows_loaded), 'normal_rows_invalid': int(normal_report.rows_invalid), 'normal_tokens_loaded': int(normal_report.tokens_loaded), 'anomaly_files_found': int(anomaly_report.files_found), 'anomaly_files_read': int(anomaly_report.files_read), 'anomaly_files_missing_feature': int(anomaly_report.files_missing_feature), 'anomaly_files_with_error': int(anomaly_report.files_with_error), 'anomaly_rows_seen': int(anomaly_report.rows_seen), 'anomaly_rows_loaded': int(anomaly_report.rows_loaded), 'anomaly_rows_invalid': int(anomaly_report.rows_invalid), 'anomaly_tokens_loaded': int(anomaly_report.tokens_loaded), 'status': 'ok', 'error': ''})
    score_path = SCORE_DIR / f'q{query_id}{FILE_TAG}_{scale_name}_{feature}_scores.npz'
    temporary = score_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, calibration=calib_scores, test_normal=normal_scores,
            test_anomaly=anom_scores, scores=test_scores, y_true=y_true,
            fit_identity=np.asarray(fit_ids), calibration_identity=np.asarray(calib_ids),
            test_identity=np.asarray(test_normal_ids + test_anom_ids),
            test_subtype=np.asarray(list(normals.subtypes[test_normal_idx]) + list(anomalies.subtypes[test_anom_idx])),
            job_signature=np.asarray(JOB_SIGNATURE))
    os.replace(temporary, score_path)
    result.update(scores_file=str(score_path), job_signature=JOB_SIGNATURE,
                  anomaly_class=ANOMALY_CLASS, k_neighbors=KNN_K)
    del Xfit
    del Xcalib
    del Xtest_normal
    del Xtest_anom
    del transform
    del model
    del calib_scores
    del normal_scores
    del anom_scores
    del test_scores
    del y_true
    del normals
    del anomalies
    gc.collect()
    return result

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f'.{path.name}.tmp')
    try:
        with temp_path.open('w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    temp_path = path.with_name(f'.{path.name}.tmp')
    try:
        with temp_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, '') for k in fields})
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

# Scientific implementation retained from Eval_Vec_Full_NoBag_s1.py.
from __future__ import annotations


import argparse


import csv


import gc


import glob


import hashlib


import json


import os


import resource


import subprocess


import sys


import time


import traceback


from dataclasses import dataclass


from datetime import datetime


from pathlib import Path


from typing import Any, Dict, List, Optional, Tuple


from zoneinfo import ZoneInfo


import numpy as np


import pyarrow.parquet as pq


from scipy.sparse import csr_matrix


from sklearn.metrics import (
    average_precision_score,
    f1_score,
    pairwise_distances_chunked,
    precision_score,
    recall_score,
    roc_auc_score,
)


CODE_VERSION = "2026-10-18-full-binary-vectors-lossless-v1"


SPLIT_POLICY = "strict_reuse_of_set_based_shared_manifest"


VECTOR_METRIC = "cosine"


RANDOM_SEED = 13


CALIB_FRAC = 0.20


THRESHOLD_PERCENTILES = [95, 99]


BOOTSTRAP_B = 1000


EMPTY_PROV_SCORE = 0.5  # Identical to the supplied set evaluator.


PARQUET_BATCH_SIZE = 10_000


PARQUET_READ_ATTEMPTS = 3


PARQUET_RETRY_DELAY_S = 2.0


PARQUET_USE_THREADS = True


PAIRWISE_WORKING_MEMORY_MB = 512


PAIRWISE_N_JOBS = -1


DERIVED_PARQUET_SUFFIXES = (".bagcompact.parquet", ".bagtfcompact.parquet")


EXCLUDED_PATH_MARKERS = ("bad_parquet", "bad_parquert")


def path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


@dataclass
class MetaRow:
    identity: str
    subtype: str


@dataclass
class RaggedSetDataset:
    tokens: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    subtypes: np.ndarray
    identities: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1


def array_nbytes(array: np.ndarray) -> int:
    total = int(array.nbytes)
    if array.dtype == object:
        total += sum(
            len(str(value).encode("utf-8"))
            for value in array
        )
    return total


def ragged_set_info(data: RaggedSetDataset) -> Dict[str, Any]:
    representation_nbytes = int(data.tokens.nbytes + data.offsets.nbytes)
    dataset_nbytes = (
        representation_nbytes
        + array_nbytes(data.labels)
        + array_nbytes(data.subtypes)
        + array_nbytes(data.identities)
    )
    return {
        "shape": str((len(data), "ragged")),
        "tokens": int(len(data.tokens)),
        "representation_kb": float(representation_nbytes / 1024.0),
        "dataset_kb": float(dataset_nbytes / 1024.0),
        "dtype": str(data.tokens.dtype),
    }


class RaggedSetBuilder:
    def __init__(self) -> None:
        self.token_chunks: List[np.ndarray] = []
        self.offsets: List[int] = [0]
        self.labels: List[int] = []
        self.subtypes: List[str] = []
        self.identities: List[str] = []

    def append(self, tokens: np.ndarray, label: int, subtype: str, identity: str) -> None:
        if len(tokens) > 0:
            self.token_chunks.append(tokens)
        self.offsets.append(self.offsets[-1] + len(tokens))
        self.labels.append(label)
        self.subtypes.append(subtype)
        self.identities.append(identity)

    def finalize(self) -> RaggedSetDataset:
        if self.token_chunks:
            tokens = np.concatenate(self.token_chunks).astype(np.int64, copy=False)
        else:
            tokens = np.empty(0, dtype=np.int64)

        return RaggedSetDataset(
            tokens=tokens,
            offsets=np.asarray(self.offsets, dtype=np.int64),
            labels=np.asarray(self.labels, dtype=np.int8),
            subtypes=np.asarray(self.subtypes, dtype=object),
            identities=np.asarray(self.identities, dtype=str),
        )


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def is_real_parquet_path(path: str | Path) -> bool:
    """Aceita apenas os Parquet principais usados pelas representações Full.

    Ignora ficheiros AppleDouble e sidecars de BAG guardados nas mesmas pastas.
    """
    parquet_path = Path(path)
    name = parquet_path.name
    lower_name = name.lower()
    lower_parts = tuple(part.lower() for part in parquet_path.parts)
    return (
        lower_name.endswith(".parquet")
        and not name.startswith("._")
        and not lower_name.endswith(DERIVED_PARQUET_SUFFIXES)
        and not any(
            marker in part
            for part in lower_parts
            for marker in EXCLUDED_PATH_MARKERS
        )
    )


def parquet_files(pattern: str | Path) -> List[str]:
    return sorted(
        f for f in glob.glob(str(pattern), recursive=True)
        if is_real_parquet_path(f)
    )


def query_dir(query: str) -> Path:
    # Ex.: q20 -> /mnt/provenance_ro/TPC-DS/Queries/Q20
    return EXTERNAL_ROOT / query.upper()


def normal_glob(query: str) -> str:
    q = query.lower()
    candidates = [
        query_dir(query) / f"{q}_normal" / "**" / "*.parquet",
        EXTERNAL_ROOT / f"{q}_normal" / "**" / "*.parquet",
    ]

    for p in candidates:
        if parquet_files(p):
            return str(p)

    return str(candidates[0])


def anomaly_glob(query: str) -> str:
    q = query.lower()
    candidates = [
        query_dir(query) / f"{q}_anomalies" / "**" / "*.parquet",
        query_dir(query) / f"{q}_annomalies" / "**" / "*.parquet",
        EXTERNAL_ROOT / f"{q}_anomalies" / "**" / "*.parquet",
        EXTERNAL_ROOT / f"{q}_annomalies" / "**" / "*.parquet",
    ]

    for p in candidates:
        if parquet_files(p):
            return str(p)

    return str(candidates[0])


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def make_identity(row: Dict[str, Any]) -> str:
    qid = _safe_int(row.get("query_id"), -1)
    qname = str(row.get("query_name", ""))
    return f"{qid}|{qname}"


def _to_sorted_unique_int_array(x: Any) -> np.ndarray:
    if x is None:
        return np.empty(0, dtype=np.int64)

    if hasattr(x, "as_py"):
        x = x.as_py()

    if x is None:
        return np.empty(0, dtype=np.int64)

    if isinstance(x, (list, tuple, set, frozenset, np.ndarray)):
        vals = [int(v) for v in x if v is not None]
    else:
        vals = [int(x)]

    if not vals:
        return np.empty(0, dtype=np.int64)

    return np.asarray(sorted(set(vals)), dtype=np.int64)


def token_column_for_method(method: str) -> str:
    if method == "vec_lineage":
        return "lineage_hash"
    if method == "vec_atomic":
        return "monoids_hash"
    raise ValueError(method)


def provenance_name_for_method(method: str) -> str:
    if method == "vec_lineage":
        return "lineage"
    if method == "vec_atomic":
        return "atomic_why"
    raise ValueError(method)


def metadata_columns() -> List[str]:
    return ["query_id", "query_name", "subtype"]


def data_columns_for_method(method: str) -> List[str]:
    return ["query_id", "query_name", "subtype", token_column_for_method(method)]


def load_metadata(pattern: str) -> List[MetaRow]:
    files = parquet_files(pattern)
    if not files:
        raise FileNotFoundError(f"Nenhum ficheiro encontrado: {pattern}")

    cols = metadata_columns()
    out: List[MetaRow] = []
    rows_seen = 0
    t0 = time.perf_counter()

    for file_i, path in enumerate(files, 1):
        schema_names = pq.read_schema(path).names
        use_cols = [c for c in cols if c in schema_names]
        missing = [c for c in cols if c not in schema_names]
        if {"query_id", "query_name"}.intersection(missing):
            raise ValueError(f"{path}: missing identity columns {missing}")

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            columns=use_cols,
            batch_size=PARQUET_BATCH_SIZE,
            use_threads=PARQUET_USE_THREADS,
        ):
            rows = batch.to_pylist()
            for row in rows:
                rows_seen += 1
                out.append(
                    MetaRow(
                        identity=make_identity(row),
                        subtype=str(row.get("subtype", "")),
                    )
                )
            del rows, batch

        print(f"  [META] {file_i}/{len(files)} rows={rows_seen:,}", flush=True)
        gc.collect()

    print(f"[META DONE] rows={len(out):,} time={time.perf_counter() - t0:.2f}s", flush=True)
    return out


SPLIT_NAMES = (
    "train_normal",
    "fit",
    "calib",
    "test_normal",
    "test_anom",
)


def split_manifest_path(query: str, scale_name: str) -> Path:
    return SPLIT_DIR / f"{query.lower()}_{scale_name}_shared_split_ids.csv"


def validate_unique_identities(meta: List[MetaRow], context: str) -> None:
    seen: set[str] = set()
    duplicates: List[str] = []

    for row in meta:
        if row.identity in seen:
            duplicates.append(row.identity)
            if len(duplicates) >= 5:
                break
        seen.add(row.identity)

    if duplicates:
        raise ValueError(
            f"{context}: identidades duplicadas; exemplos={duplicates}. "
            "A identidade query_id|query_name tem de ser única."
        )


def read_split_manifest(
    path: Path,
    query: str,
    scale_name: str,
) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifesto de splits não encontrado: {path}")

    positioned: Dict[str, List[Tuple[int, str]]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"query", "scale", "split", "pos", "identity"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: faltam colunas {sorted(missing)}")

        for row in reader:
            row_query = str(row["query"]).lower()
            row_scale = str(row["scale"])
            split_name = str(row["split"])

            if row_query != query.lower() or row_scale != scale_name:
                raise ValueError(
                    f"{path}: linha de outro contexto "
                    f"query={row_query} scale={row_scale}"
                )
            if split_name not in positioned:
                raise ValueError(f"{path}: split desconhecido {split_name!r}")

            identity = str(row["identity"])
            if not identity or "|" not in identity:
                raise ValueError(f"{path}: invalid identity {identity!r}")
            expected_label = 1 if split_name == "test_anom" else 0
            if "label" in row and int(row["label"]) != expected_label:
                raise ValueError(f"{path}: wrong label for {split_name}/{identity}")
            positioned[split_name].append((int(row["pos"]), identity))

    manifest: Dict[str, List[str]] = {}
    for split_name, rows in positioned.items():
        rows.sort(key=lambda item: item[0])
        if [pos for pos, _ in rows] != list(range(len(rows))):
            raise ValueError(f"{path}: non-contiguous or duplicate positions in {split_name}")
        identities = [identity for _, identity in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"{path}: identidades repetidas em {split_name}")
        manifest[split_name] = identities

    return manifest


def validate_split_manifest(
    manifest: Dict[str, List[str]],
    normal_meta: List[MetaRow],
    anomaly_meta: List[MetaRow],
    train_k: int,
    test_normal_k: int,
    n_anom_test: int,
) -> None:
    normal_pool = {row.identity for row in normal_meta}
    anomaly_pool = {row.identity for row in anomaly_meta}

    fit = set(manifest["fit"])
    calib = set(manifest["calib"])
    test_normal = set(manifest["test_normal"])
    test_anom = set(manifest["test_anom"])
    train_normal = set(manifest["train_normal"])

    if fit & calib or fit & test_normal or calib & test_normal:
        raise ValueError("O manifesto contém sobreposição entre splits normais.")
    if train_normal != fit | calib:
        raise ValueError("train_normal tem de ser exatamente a união de fit e calib.")
    if not (train_normal | test_normal) <= normal_pool:
        missing = (train_normal | test_normal).difference(normal_pool)
        raise ValueError(f"Execuções normais ausentes do Parquet: {sorted(missing)[:5]}")
    if not (test_anom <= anomaly_pool):
        missing = test_anom.difference(anomaly_pool)
        raise ValueError(f"Anomalias ausentes do Parquet: {sorted(missing)[:5]}")

    expected_train = train_k
    expected_test_normal = test_normal_k

    expected_counts = {
        "train_normal": expected_train,
        "fit": expected_train - max(1, int(train_k * CALIB_FRAC)),
        "calib": max(1, int(train_k * CALIB_FRAC)),
        "test_normal": expected_test_normal,
        "test_anom": n_anom_test,
    }

    for split_name, expected in expected_counts.items():
        actual = len(manifest[split_name])
        if actual != expected:
            raise ValueError(
                f"Split {split_name}: manifesto={actual:,}, esperado={expected:,}"
            )


def load_required_split_manifest(
    query: str,
    scale_name: str,
    normal_meta: List[MetaRow],
    anomaly_meta: List[MetaRow],
    train_k: int,
    test_normal_k: int,
    n_anom_test: int,
) -> Tuple[Dict[str, List[str]], Path]:
    validate_unique_identities(normal_meta, f"{query}/normal")
    validate_unique_identities(anomaly_meta, f"{query}/anomaly")

    path = split_manifest_path(query, scale_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Split partilhado obrigatório não encontrado: {path}. "
            "Este avaliador vetorial nunca cria splits: deve reutilizar "
            "exatamente os manifests da experiência set-based."
        )

    print(f"[SPLIT REUSE REQUIRED] {path}", flush=True)

    manifest = read_split_manifest(path, query=query, scale_name=scale_name)
    validate_split_manifest(
        manifest=manifest,
        normal_meta=normal_meta,
        anomaly_meta=anomaly_meta,
        train_k=train_k,
        test_normal_k=test_normal_k,
        n_anom_test=n_anom_test,
    )
    return manifest, path


def read_file_identity_matches(
    path: str,
    selected_identities: set[str],
) -> Tuple[int, set[str]]:
    identity_columns = ["query_id", "query_name"]
    last_error: Optional[OSError] = None

    for attempt in range(1, PARQUET_READ_ATTEMPTS + 1):
        use_threads = PARQUET_USE_THREADS if attempt == 1 else False
        try:
            parquet_file = pq.ParquetFile(path)
            schema_names = set(parquet_file.schema_arrow.names)
            missing = set(identity_columns).difference(schema_names)
            if missing:
                raise ValueError(
                    f"{path}: faltam colunas de identidade {sorted(missing)}"
                )

            rows_seen = 0
            matches: set[str] = set()
            for batch in parquet_file.iter_batches(
                columns=identity_columns,
                batch_size=PARQUET_BATCH_SIZE,
                use_threads=use_threads,
            ):
                rows = batch.to_pylist()
                rows_seen += len(rows)
                for row in rows:
                    identity = make_identity(row)
                    if identity in selected_identities:
                        matches.add(identity)
                del rows, batch

            return rows_seen, matches
        except OSError as exc:
            last_error = exc
            print(
                f"[PARQUET READ RETRY] stage=identity attempt={attempt}/"
                f"{PARQUET_READ_ATTEMPTS} threads={use_threads} "
                f"path={path} error={exc}",
                flush=True,
            )
            if attempt < PARQUET_READ_ATTEMPTS:
                gc.collect()
                time.sleep(PARQUET_RETRY_DELAY_S * attempt)

    assert last_error is not None
    raise OSError(
        getattr(last_error, "errno", 5),
        f"Falha persistente ao ler identidades do Parquet: {path}. "
        f"Tentativas={PARQUET_READ_ATTEMPTS}. Erro original: {last_error}",
    ) from last_error


def read_selected_rows_from_file(
    path: str,
    method: str,
    label: int,
    file_targets: set[str],
) -> List[Tuple[str, np.ndarray, str]]:
    cols = data_columns_for_method(method)
    token_col = token_column_for_method(method)
    required_columns = {"query_id", "query_name", token_col}
    last_error: Optional[OSError] = None

    for attempt in range(1, PARQUET_READ_ATTEMPTS + 1):
        use_threads = PARQUET_USE_THREADS if attempt == 1 else False
        pending_rows: List[Tuple[str, np.ndarray, str]] = []
        pending_identities: set[str] = set()

        try:
            parquet_file = pq.ParquetFile(path)
            schema_names = set(parquet_file.schema_arrow.names)
            missing_required = required_columns.difference(schema_names)
            if missing_required:
                raise ValueError(
                    f"{path}: faltam colunas obrigatórias "
                    f"{sorted(missing_required)} para {method}"
                )
            use_cols = [column for column in cols if column in schema_names]

            for batch in parquet_file.iter_batches(
                columns=use_cols,
                batch_size=PARQUET_BATCH_SIZE,
                use_threads=use_threads,
            ):
                rows = batch.to_pylist()
                for row in rows:
                    identity = make_identity(row)
                    if (
                        identity not in file_targets
                        or identity in pending_identities
                    ):
                        continue

                    pending_rows.append(
                        (
                            identity,
                            _to_sorted_unique_int_array(row.get(token_col)),
                            str(row.get("subtype", "")),
                        )
                    )
                    pending_identities.add(identity)
                    if len(pending_identities) == len(file_targets):
                        break

                del rows, batch
                if len(pending_identities) == len(file_targets):
                    break

            missing_identities = file_targets.difference(pending_identities)
            if missing_identities:
                raise ValueError(
                    f"{path}: não foi possível carregar "
                    f"{len(missing_identities):,} identidades seleccionadas "
                    f"para {method}; exemplos={sorted(missing_identities)[:5]}"
                )

            return pending_rows
        except OSError as exc:
            last_error = exc
            pending_rows.clear()
            pending_identities.clear()
            print(
                f"[PARQUET READ RETRY] stage=payload column={token_col} "
                f"attempt={attempt}/{PARQUET_READ_ATTEMPTS} "
                f"threads={use_threads} path={path} error={exc}",
                flush=True,
            )
            if attempt < PARQUET_READ_ATTEMPTS:
                gc.collect()
                time.sleep(PARQUET_RETRY_DELAY_S * attempt)

    assert last_error is not None
    raise OSError(
        getattr(last_error, "errno", 5),
        f"Falha persistente ao ler o payload {token_col} do Parquet: {path}. "
        f"Tentativas={PARQUET_READ_ATTEMPTS}. Erro original: {last_error}",
    ) from last_error


def load_selected_set_dataset(
    pattern: str,
    method: str,
    label: int,
    selected_identities: set[str],
) -> RaggedSetDataset:
    files = parquet_files(pattern)
    if not files:
        raise FileNotFoundError(f"Nenhum ficheiro encontrado: {pattern}")

    token_col = token_column_for_method(method)
    builder = RaggedSetBuilder()

    rows_seen = 0
    rows_loaded = 0
    loaded_identities: set[str] = set()
    target = len(selected_identities)
    t0 = time.perf_counter()

    for file_i, path in enumerate(files, 1):
        if rows_loaded >= target:
            break

        remaining = selected_identities.difference(loaded_identities)
        file_rows_seen, file_targets = read_file_identity_matches(
            path=path,
            selected_identities=remaining,
        )
        rows_seen += file_rows_seen

        if file_targets:
            print(
                f"  [DATA FILE] {file_i}/{len(files)} "
                f"targets={len(file_targets):,} column={token_col} path={path}",
                flush=True,
            )
            pending_rows = read_selected_rows_from_file(
                path=path,
                method=method,
                label=label,
                file_targets=file_targets,
            )
            for identity, tokens, subtype in pending_rows:
                builder.append(
                    tokens=tokens,
                    label=label,
                    subtype=subtype,
                    identity=identity,
                )
                loaded_identities.add(identity)
                rows_loaded += 1
            del pending_rows

        print(
            f"  [DATA] {file_i}/{len(files)} rows_seen={rows_seen:,} "
            f"rows_loaded={rows_loaded:,}/{target:,}",
            flush=True,
        )
        gc.collect()

    data = builder.finalize()
    missing_selected = selected_identities.difference(loaded_identities)
    if len(data) != target or missing_selected:
        raise ValueError(
            f"Carregamento incompleto para {method}: "
            f"loaded={len(data):,} target={target:,}; "
            f"missing={len(missing_selected):,}; "
            f"examples={sorted(missing_selected)[:5]}"
        )

    print(
        f"[DATA DONE] label={label} rows={len(data):,} tokens={len(data.tokens):,} "
        f"time={time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return data


def build_explicit_vocabulary(
    datasets: List[RaggedSetDataset],
) -> np.ndarray:
    """Create one lossless coordinate for every selected provenance element.

    The vocabulary is the union of the elements in fit, calibration, normal
    test, and anomalous test. It is only a deterministic coordinate map: no
    label, count, score, or fitted statistic is used. Consequently, no element
    is dropped merely because it first appears outside the fitting subset.
    """
    vocabulary = np.empty(0, dtype=np.int64)
    for dataset in datasets:
        if len(dataset.tokens) == 0:
            continue
        local = np.unique(dataset.tokens)
        vocabulary = np.union1d(vocabulary, local).astype(np.int64, copy=False)
        del local
        gc.collect()

    if len(vocabulary) > np.iinfo(np.int32).max:
        raise MemoryError(
            f"Vocabulário explícito demasiado grande: {len(vocabulary):,} coordenadas."
        )
    return vocabulary


def ragged_set_to_binary_csr(
    dataset: RaggedSetDataset,
    vocabulary: np.ndarray,
) -> csr_matrix:
    """Encode each set exactly as a binary row in a shared vector space."""
    if len(dataset.tokens) == 0:
        return csr_matrix(
            (len(dataset), len(vocabulary)),
            dtype=np.float32,
        )

    positions = np.searchsorted(vocabulary, dataset.tokens)
    in_bounds = positions < len(vocabulary)
    if not bool(np.all(in_bounds)):
        raise RuntimeError("A vectorização encontrou elementos fora do vocabulário.")
    if not bool(np.all(vocabulary[positions] == dataset.tokens)):
        raise RuntimeError("A vectorização perdeu elementos de proveniência.")

    row_sizes = np.diff(dataset.offsets)
    if (dataset.offsets[0] != 0 or np.any(row_sizes < 0)
            or int(row_sizes.sum()) != len(dataset.tokens)):
        raise RuntimeError("Offsets inválidos no conjunto de proveniência.")

    matrix = csr_matrix(
        (
            np.ones(len(positions), dtype=np.float32),
            positions.astype(np.int32, copy=False),
            dataset.offsets.astype(np.int64, copy=True),
        ),
        shape=(len(dataset), len(vocabulary)),
        dtype=np.float32,
    )
    matrix.sort_indices()
    if not matrix.has_canonical_format:
        raise RuntimeError("Duplicate coordinates in a supposedly deduplicated set.")
    if not np.all(matrix.data == 1):
        raise RuntimeError("Non-binary coordinates.")

    if matrix.nnz != len(dataset.tokens):
        raise RuntimeError(
            "A matriz binária não preservou exatamente a cardinalidade dos sets."
        )
    if not np.array_equal(np.diff(matrix.indptr), row_sizes):
        raise RuntimeError(
            "A matriz binária não preservou a cardinalidade de pelo menos uma execução."
        )
    return matrix


def matrix_info(matrix: csr_matrix) -> Dict[str, Any]:
    nbytes = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return {
        "shape": str(matrix.shape),
        "nnz": int(matrix.nnz),
        "kb": float(nbytes / 1024.0),
        "dtype": str(matrix.dtype),
    }


def score_cosine_1nn_chunked(
    fit: csr_matrix,
    data: csr_matrix,
    label: str,
) -> Tuple[np.ndarray, float]:
    if K_NEIGHBORS not in (1, 5):
        raise ValueError("Supported neighbourhood sizes: 1 and 5")
    if fit.shape[0] == 0:
        raise ValueError("O fitting não contém execuções.")

    started = time.perf_counter()
    if data.shape[0] == 0:
        return np.empty(0, dtype=np.float32), time.perf_counter() - started
    if fit.shape[1] == 0:
        return np.full(data.shape[0], EMPTY_PROV_SCORE, dtype=np.float32), time.perf_counter() - started
    parts: List[np.ndarray] = []
    rows_done = 0
    for distances in pairwise_distances_chunked(
        data,
        fit,
        metric=VECTOR_METRIC,
        n_jobs=PAIRWISE_N_JOBS,
        working_memory=PAIRWISE_WORKING_MEMORY_MB,
    ):
        if K_NEIGHBORS == 1:
            nearest = np.min(distances, axis=1).astype(np.float32, copy=False)
        else:
            k = min(K_NEIGHBORS, fit.shape[0])
            top = np.partition(distances, k - 1, axis=1)[:, :k]
            nearest = np.mean(top, axis=1, dtype=np.float64).astype(np.float32)
        # Empty EVALUATION sets have the same fixed score as the set reference.
        empty = np.diff(data.indptr[rows_done:rows_done + len(nearest) + 1]) == 0
        nearest[empty] = EMPTY_PROV_SCORE
        if not np.all(np.isfinite(nearest)):
            raise RuntimeError("Non-finite cosine scores.")
        parts.append(nearest)
        rows_done += len(nearest)
        print(
            f"  [SCORE CHUNK] {label} rows={rows_done:,}/{data.shape[0]:,}",
            flush=True,
        )
        del distances, nearest
        gc.collect()

    scores = (
        np.concatenate(parts).astype(np.float32, copy=False)
        if parts
        else np.empty(0, dtype=np.float32)
    )
    return scores, float(time.perf_counter() - started)


def compute_threshold_bootstrap(calib_scores: np.ndarray, percentile: int, B: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    thresholds = np.zeros(B, dtype=np.float64)
    for b in range(B):
        sample = rng.choice(calib_scores, size=len(calib_scores), replace=True)
        thresholds[b] = np.percentile(sample, percentile)
    return float(np.mean(thresholds)), float(np.percentile(thresholds, 2.5)), float(np.percentile(thresholds, 97.5))


def safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan")


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def eval_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores > threshold).astype(np.int8)
    return {
        "ap": safe_ap(y_true, scores),
        "auc": safe_auc(y_true, scores),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n_pred": int(y_pred.sum()),
    }


def run_one(query: str, scale: dict, method: str) -> Dict[str, Any]:
    scale_name = str(scale["name"])
    train_k = int(scale["train_k"])
    test_normal_k = int(scale["test_normal_k"])
    n_anom_test = int(scale["n_anom_test"])

    n_glob = normal_glob(query)
    a_glob = anomaly_glob(query)

    print("\n" + "#" * 100)
    print(f"[QUERY] {query.upper()} | [SCALE] {scale_name} | [METHOD] {method}")
    print("#" * 100)
    print(f"[PATH] normals={n_glob}")
    print(f"[PATH] anomalies={a_glob}")
    print(f"[MODE] full explicit set-to-vector | cosine {K_NEIGHBORS}-NN | no hashing | no SVD")
    print("[SPLIT POLICY] reuse the exact set-based split manifest")
    print(f"[SAFE] parquet_use_threads={PARQUET_USE_THREADS}")

    total_start = time.perf_counter()
    job_signature = current_job_signature(query, scale_name)

    print("[PASS 1] load normal metadata")
    metadata_start = time.perf_counter()
    normal_meta = load_metadata(n_glob)
    print("[PASS 1] load anomaly metadata")
    anomaly_meta = load_metadata(a_glob)
    metadata_load_s = time.perf_counter() - metadata_start

    manifest_start = time.perf_counter()
    manifest, manifest_path = load_required_split_manifest(
        query=query,
        scale_name=scale_name,
        normal_meta=normal_meta,
        anomaly_meta=anomaly_meta,
        train_k=train_k,
        test_normal_k=test_normal_k,
        n_anom_test=n_anom_test,
    )
    manifest_s = time.perf_counter() - manifest_start

    fit_ids = set(manifest["fit"])
    calib_ids = set(manifest["calib"])
    test_normal_ids = set(manifest["test_normal"])
    test_anom_ids = set(manifest["test_anom"])
    n_train_original = len(manifest["train_normal"])

    del manifest, normal_meta, anomaly_meta
    gc.collect()

    print(
        f"[SPLIT NORMAL] train={n_train_original:,} fit={len(fit_ids):,} "
        f"calib={len(calib_ids):,} test_normal={len(test_normal_ids):,}",
        flush=True,
    )
    print(f"[SPLIT ANOM] test_anom={len(test_anom_ids):,}", flush=True)

    print("[PASS 2] load fit set elements")
    load_fit_start = time.perf_counter()
    fit = load_selected_set_dataset(n_glob, method, 0, fit_ids)
    load_fit_s = time.perf_counter() - load_fit_start
    fit_info = ragged_set_info(fit)
    del fit_ids
    gc.collect()
    print(f"[FIT] rows={len(fit):,} tokens={len(fit.tokens):,}", flush=True)

    print("[PASS 2] load calibration set elements")
    load_calib_start = time.perf_counter()
    calib = load_selected_set_dataset(n_glob, method, 0, calib_ids)
    load_calib_s = time.perf_counter() - load_calib_start
    calib_info = ragged_set_info(calib)
    del calib_ids
    gc.collect()

    n_calib = len(calib)

    print("[PASS 2] load normal-test set elements")
    load_normal_start = time.perf_counter()
    test_normal = load_selected_set_dataset(n_glob, method, 0, test_normal_ids)
    load_normal_s = time.perf_counter() - load_normal_start
    test_normal_info = ragged_set_info(test_normal)
    del test_normal_ids
    gc.collect()

    n_test_normal = len(test_normal)

    print("[PASS 2] load anomalous-test set elements")
    load_anom_start = time.perf_counter()
    test_anom = load_selected_set_dataset(a_glob, method, 1, test_anom_ids)
    load_anom_s = time.perf_counter() - load_anom_start
    test_anom_info = ragged_set_info(test_anom)
    del test_anom_ids
    gc.collect()

    n_anom = len(test_anom)

    audit_arrays = {}
    for name, dataset in [
        ("fit", fit), ("calib", calib),
        ("test_normal", test_normal), ("test_anom", test_anom),
    ]:
        audit_arrays[f"{name}_identities"] = dataset.identities
        audit_arrays[f"{name}_subtypes"] = dataset.subtypes.astype(str)
        audit_arrays[f"{name}_cardinalities"] = np.diff(dataset.offsets).copy()

    load_s = (
        metadata_load_s
        + load_fit_s
        + load_calib_s
        + load_normal_s
        + load_anom_s
    )

    print("[VECTOR] build one-to-one vocabulary over all selected sets", flush=True)
    build_start = time.perf_counter()
    vocabulary = build_explicit_vocabulary(
        [fit, calib, test_normal, test_anom]
    )
    vocabulary_build_s = time.perf_counter() - build_start
    print(
        f"[VECTOR] dimensions={len(vocabulary):,} "
        f"vocabulary={vocabulary.nbytes / 1024.0:.2f} KiB",
        flush=True,
    )

    matrix_start = time.perf_counter()
    Xfit = ragged_set_to_binary_csr(fit, vocabulary)
    Xfit_info = matrix_info(Xfit)
    del fit
    gc.collect()

    Xcalib = ragged_set_to_binary_csr(calib, vocabulary)
    Xcalib_info = matrix_info(Xcalib)
    del calib
    gc.collect()

    Xtest_normal = ragged_set_to_binary_csr(test_normal, vocabulary)
    Xtest_normal_info = matrix_info(Xtest_normal)
    del test_normal
    gc.collect()

    Xtest_anom = ragged_set_to_binary_csr(test_anom, vocabulary)
    Xtest_anom_info = matrix_info(Xtest_anom)
    del test_anom
    gc.collect()

    matrix_build_s = time.perf_counter() - matrix_start
    representation_build_s = vocabulary_build_s + matrix_build_s
    print(
        f"[VECTOR DONE] Xfit={Xfit.shape} nnz={Xfit.nnz:,} "
        f"build={representation_build_s:.2f}s",
        flush=True,
    )

    print(f"[SCORE] calibration cosine {K_NEIGHBORS}-NN", flush=True)
    calib_scores, score_calib_s = score_cosine_1nn_chunked(
        Xfit, Xcalib, "calib"
    )
    del Xcalib
    gc.collect()

    print(f"[SCORE] normal test cosine {K_NEIGHBORS}-NN", flush=True)
    normal_scores, score_normal_s = score_cosine_1nn_chunked(
        Xfit, Xtest_normal, "test_normal"
    )
    del Xtest_normal
    gc.collect()

    print(f"[SCORE] anomalous test cosine {K_NEIGHBORS}-NN", flush=True)
    anom_scores, score_anom_s = score_cosine_1nn_chunked(
        Xfit, Xtest_anom, "test_anom"
    )
    del Xtest_anom, Xfit
    gc.collect()

    test_scores = np.concatenate([normal_scores, anom_scores]).astype(np.float32, copy=False)
    y_true = np.concatenate([
        np.zeros(len(normal_scores), dtype=np.int8),
        np.ones(len(anom_scores), dtype=np.int8),
    ])

    metrics_start = time.perf_counter()
    base = eval_scores(y_true=y_true, scores=test_scores, threshold=np.inf)
    contamination = float(n_anom / max(1, len(test_scores)))
    score_total_s = score_calib_s + score_normal_s + score_anom_s
    representation_total_kb = (
        Xfit_info["kb"]
        + Xcalib_info["kb"]
        + Xtest_normal_info["kb"]
        + Xtest_anom_info["kb"]
        + vocabulary.nbytes / 1024.0
    )

    result: Dict[str, Any] = {
        "query": query,
        "method": "knn_distance",
        "feature": method,
        "scale": scale_name,
        "anomaly_class": "s1",
        "job_signature": job_signature,
        "shared_split_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "code_version": CODE_VERSION,
        "code_sha256": script_sha256(),
        "split_policy": SPLIT_POLICY,
        "representation": (
            "full_sparse_binary_lineage_vector"
            if method == "vec_lineage"
            else "full_sparse_binary_atomic_why_vector"
        ),
        "representation_family": "explicit_sparse_binary_vector",
        "representation_dim": int(len(vocabulary)),
        "input_set_column": token_column_for_method(method),
        "base_provenance": provenance_name_for_method(method),
        "vector_metric": VECTOR_METRIC,
        "uses_feature_hashing": False,
        "uses_buckets": False,
        "uses_svd": False,
        "uses_reduction": False,
        "lossless_set_to_vector": True,
        "vocabulary_scope": "union_of_all_selected_splits_labels_ignored",
        "candidate_policy": f"cosine_{K_NEIGHBORS}nn_against_all_fit",
        "train_k": train_k,
        "test_normal_k": test_normal_k,
        "n_anom_test_config": n_anom_test,
        "n_train": n_train_original,
        "n_fit": n_train_original - n_calib,
        "n_fit_actual": n_train_original - n_calib,
        "n_calib": n_calib,
        "n_test": len(test_scores),
        "n_test_normal": n_test_normal,
        "n_anom": n_anom,
        "n_test_anom": n_anom,
        "contamination": contamination,
        "k_neighbors": K_NEIGHBORS,
        "empty_prov_score": EMPTY_PROV_SCORE,
        "distance_metric": VECTOR_METRIC,
        "uses_hashing_vectorizer": False,
        "uses_kmv": False,
        "uses_countsketch": False,
        "uses_tfidf": False,
        "discarded_elements": 0,
        "scoring_order": "sorted_parquet_paths_original_row_order_same_as_set_evaluator",
        "bootstrap_b": BOOTSTRAP_B,
        "threshold_percentiles": ",".join(map(str, THRESHOLD_PERCENTILES)),
        "calib_frac": CALIB_FRAC,
        "parquet_batch_size": PARQUET_BATCH_SIZE,
        "parquet_use_threads": PARQUET_USE_THREADS,
        "parquet_read_attempts": PARQUET_READ_ATTEMPTS,
        "parquet_retry_delay_s": PARQUET_RETRY_DELAY_S,
        "input_root": str(EXTERNAL_ROOT),
        "output_dir": str(OUT_DIR),
        "split_manifest": str(manifest_path),
        "uses_shared_split": True,
        "shared_split_dir": str(SPLIT_DIR),
        "shared_split_file": str(manifest_path),
        "tmp_dir": str(TMP_DIR),
        "load_protocol": "selected_full_sets_then_lossless_shared_binary_coordinates",
        "quality_protocol": "shared_split_threshold_from_calib_percentile",
        "cost_protocol": f"parquet_load_plus_explicit_csr_build_plus_chunked_cosine_{K_NEIGHBORS}nn",
        "runtime_comparison_mode": f"full_explicit_sparse_vector_cosine_{K_NEIGHBORS}nn",
        "comparison_ready": True,
        "vocabulary_size": int(len(vocabulary)),
        "vocabulary_kb": float(vocabulary.nbytes / 1024.0),
        "source_fit_tokens": fit_info["tokens"],
        "source_calib_tokens": calib_info["tokens"],
        "source_test_normal_tokens": test_normal_info["tokens"],
        "source_test_anom_tokens": test_anom_info["tokens"],
        "xfit_shape": Xfit_info["shape"],
        "xfit_nnz": Xfit_info["nnz"],
        "xfit_kb": Xfit_info["kb"],
        "xfit_dtype": Xfit_info["dtype"],
        "xcalib_shape": Xcalib_info["shape"],
        "xcalib_nnz": Xcalib_info["nnz"],
        "xcalib_kb": Xcalib_info["kb"],
        "xtest_normal_shape": Xtest_normal_info["shape"],
        "xtest_normal_nnz": Xtest_normal_info["nnz"],
        "xtest_normal_kb": Xtest_normal_info["kb"],
        "xtest_anom_shape": Xtest_anom_info["shape"],
        "xtest_anom_nnz": Xtest_anom_info["nnz"],
        "xtest_anom_kb": Xtest_anom_info["kb"],
        "xtest_shape": str((n_test_normal + n_anom, len(vocabulary))),
        "xtest_nnz": int(Xtest_normal_info["nnz"] + Xtest_anom_info["nnz"]),
        "xtest_kb": float(Xtest_normal_info["kb"] + Xtest_anom_info["kb"]),
        "representation_total_kb": representation_total_kb,
        "ap": base["ap"],
        "auc": base["auc"],
        "calib_score_min": float(np.min(calib_scores)),
        "calib_score_median": float(np.median(calib_scores)),
        "calib_score_max": float(np.max(calib_scores)),
        "test_score_min": float(np.min(test_scores)),
        "test_score_median": float(np.median(test_scores)),
        "test_score_max": float(np.max(test_scores)),
        "score_calib_s": score_calib_s,
        "score_normal_s": score_normal_s,
        "score_anom_s": score_anom_s,
        "score_s": score_total_s,
        "score_total_s": score_total_s,
        "model_score_s": score_total_s,
        "metadata_load_s": metadata_load_s,
        "manifest_s": manifest_s,
        "load_fit_s": load_fit_s,
        "load_calib_s": load_calib_s,
        "load_normal_s": load_normal_s,
        "load_anom_s": load_anom_s,
        "load_s": load_s,
        "vocabulary_build_s": vocabulary_build_s,
        "matrix_build_s": matrix_build_s,
        "representation_build_s": representation_build_s,
        "fit_s": 0.0,
        "preprocess_s": load_s + representation_build_s,
        "preprocess_total_s": load_s + representation_build_s,
    }

    for pct in THRESHOLD_PERCENTILES:
        thr_mean, ci_lo, ci_hi = compute_threshold_bootstrap(calib_scores, pct, BOOTSTRAP_B, RANDOM_SEED)
        metrics = eval_scores(y_true=y_true, scores=test_scores, threshold=thr_mean)
        suffix = f"_pct{pct}"
        result[f"threshold{suffix}"] = metrics["threshold"]
        result[f"thr_ci_lo{suffix}"] = ci_lo
        result[f"thr_ci_hi{suffix}"] = ci_hi
        result[f"f1{suffix}"] = metrics["f1"]
        result[f"precision{suffix}"] = metrics["precision"]
        result[f"recall{suffix}"] = metrics["recall"]
        result[f"n_pred{suffix}"] = metrics["n_pred"]
        print(
            f"    pct{pct}: thr={thr_mean:.6f} f1={metrics['f1']:.4f} "
            f"p={metrics['precision']:.4f} r={metrics['recall']:.4f} n_pred={metrics['n_pred']}",
            flush=True,
        )

    result["metrics_s"] = float(time.perf_counter() - metrics_start)
    result["total_s"] = float(load_s + representation_build_s + score_total_s)
    result["end_to_end_s"] = float(time.perf_counter() - total_start)
    result["runtime_s"] = result["total_s"]
    result["peak_rss_mb"] = float(peak_rss_mb())
    if current_job_signature(query, scale_name) != job_signature:
        raise RuntimeError("Inputs or split changed during evaluation; result not published.")
    result["status"] = "ok"
    result["error"] = ""

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    score_path = SCORE_DIR / f"{query}_{scale_name}_{method}_scores.npz"
    artifact_start = time.perf_counter()
    np.savez_compressed(
        score_path, **audit_arrays,
        calib_scores=calib_scores, normal_scores=normal_scores,
        anom_scores=anom_scores, test_scores=test_scores, y_true=y_true,
        code_version=np.asarray(CODE_VERSION),
        job_signature=np.asarray(job_signature),
        split_sha256=np.asarray(result["shared_split_sha256"]),
        vocabulary_sha256=np.asarray(hashlib.sha256(vocabulary.tobytes()).hexdigest()),
    )
    result["scores_file"] = str(score_path)
    result["scores_sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
    result["artifact_write_s"] = time.perf_counter() - artifact_start

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{query}_vector_full_explicit_{method}_{scale_name}.csv"
    all_keys = sorted(result.keys())

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in all_keys})

    print(
        f"[OK] guardado -> {out_csv} ap={result['ap']:.4f} "
        f"auc={result['auc']:.4f} time={result['runtime_s']:.2f}s",
        flush=True,
    )


    del vocabulary, calib_scores, normal_scores, anom_scores, test_scores, y_true
    gc.collect()
    return result


def run_vectorization_self_test() -> None:
    def dataset(rows: List[List[int]]) -> RaggedSetDataset:
        builder = RaggedSetBuilder()
        for i, row in enumerate(rows):
            builder.append(_to_sorted_unique_int_array(row), 0, "synthetic", f"{i}|synthetic")
        return builder.finalize()

    fit = dataset([[1, 2], []])
    evaluation = dataset([[1, 3], [4], [], [1, 2, 2]])
    vocab = build_explicit_vocabulary([fit, evaluation])
    xf = ragged_set_to_binary_csr(fit, vocab)
    xe = ragged_set_to_binary_csr(evaluation, vocab)
    if vocab.tolist() != [1, 2, 3, 4] or xe.nnz != 5:
        raise RuntimeError("Self-test: membership loss.")
    scores, _ = score_cosine_1nn_chunked(xf, xe, "self-test")
    np.testing.assert_allclose(scores, [0.5, 1.0, EMPTY_PROV_SCORE, 0.0], atol=2e-7)
    try:
        ragged_set_to_binary_csr(evaluation, np.asarray([1, 2], dtype=np.int64))
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Self-test: unseen identifiers were silently dropped.")
    print("[SELF-TEST] full membership, unseen identifiers, cosine and empty sets: PASS", flush=True)


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0 * 1024.0)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def scale_by_name(name: str) -> Dict[str, Any]:
    for scale in SCALES:
        if scale["name"] == name:
            return scale
    raise KeyError(name)


def current_job_signature(
    query: str,
    scale_name: str,
    *,
    code_sha256: Optional[str] = None,
) -> str:
    split = split_manifest_path(query, scale_name)
    if not split.is_file():
        raise FileNotFoundError(f"Required existing split: {split}; no split will be created.")
    inputs = []
    for pattern in [normal_glob(query), anomaly_glob(query)]:
        paths = parquet_files(pattern)
        if not paths:
            raise FileNotFoundError(f"No full Parquets: {pattern}")
        for path in paths:
            stat = Path(path).stat()
            inputs.append((str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns))
    payload = {
        "code_sha256": script_sha256() if code_sha256 is None else code_sha256,
        "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "input_files_size_mtime": inputs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def resource_failure_reason(row: Dict[str, Any]) -> Optional[str]:
    if row.get("status") == "ok":
        return None
    if row.get("status") == "timeout" or row.get("error_type") == "TimeoutExpired":
        return "timeout"
    if row.get("status") in {"oom_or_sigkill", "oom", "out_of_memory", "memory_error"}:
        return "memory_or_sigkill"
    if row.get("returncode") in (-9, "-9"):
        return "memory_or_sigkill"
    error_type = str(row.get("error_type", "")).rsplit(".", 1)[-1].lower()
    if error_type in {
        "memoryerror", "_arraymemoryerror", "arrowmemoryerror",
        "outofmemoryerror", "outofmemoryexception",
    }:
        return "memory"
    details = (str(row.get("error", "")) + "\n" + str(row.get("traceback", ""))).lower()
    if any(marker in details for marker in (
        "memoryerror", "out of memory", "out-of-memory", "cannot allocate memory",
        "unable to allocate", "failed to allocate memory", "std::bad_alloc",
    )):
        return "memory"
    if error_type == "terminatedworkererror" and "sigkill" in details:
        return "memory_or_sigkill"
    return None



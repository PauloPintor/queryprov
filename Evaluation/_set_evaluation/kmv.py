# Scientific implementation retained from Set_KMV.py.
from __future__ import annotations


import argparse


import ast


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


from collections import defaultdict


from dataclasses import dataclass


from pathlib import Path


from typing import Any, Dict, List, Optional, Tuple


import numpy as np


import pyarrow.parquet as pq


from numba import njit, prange


from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


CODE_VERSION = "2026-09-25-kmv-s1-s2-unified-v1"


DERIVED_PARQUET_SUFFIXES = (
    ".bagcompact.parquet",
    ".bagtfcompact.parquet",
)


EXCLUDED_PATH_MARKERS = (
    "bad_parquet",
    "bad_parquert",
)


RANDOM_SEED = 13


CALIB_FRAC = 0.20


THRESHOLD_PERCENTILES = [95, 99]


BOOTSTRAP_B = 1000


EMPTY_PROV_SCORE = 0.5


PARQUET_BATCH_SIZE = 10_000


PARQUET_USE_THREADS = True


METHOD_KMV_COLS = {
    "lineage_jaccard": {
        512: [
            "lineage_kmv_512",
            "lineage_hash_kmv_512",
            "lineage_512",
            "kmv_512_lineage",
        ],
        1024: [
            "lineage_kmv_1024",
            "lineage_hash_kmv_1024",
            "lineage_1024",
            "kmv_1024_lineage",
        ],
    },
    "atomic_why_jaccard": {
        512: [
            "monoids_kmv_512",
            "monoids_hash_kmv_512",
            "monoids_512",
            "kmv_512_monoids",
        ],
        1024: [
            "monoids_kmv_1024",
            "monoids_hash_kmv_1024",
            "monoids_1024",
            "kmv_1024_monoids",
        ],
    },
}


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
class RaggedKMVDataset:
    hashes: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray
    subtypes: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1


class RaggedKMVBuilder:
    def __init__(self) -> None:
        self.hash_chunks: List[np.ndarray] = []
        self.offsets: List[int] = [0]
        self.labels: List[int] = []
        self.subtypes: List[str] = []

    def append(
        self,
        hashes: np.ndarray,
        label: int,
        subtype: str,
    ) -> None:
        if len(hashes) > 0:
            self.hash_chunks.append(hashes)

        self.offsets.append(self.offsets[-1] + len(hashes))
        self.labels.append(label)
        self.subtypes.append(subtype)

    def finalize(self) -> RaggedKMVDataset:
        if self.hash_chunks:
            hashes = np.concatenate(self.hash_chunks).astype(np.uint64, copy=False)
        else:
            hashes = np.empty(0, dtype=np.uint64)

        return RaggedKMVDataset(
            hashes=hashes,
            offsets=np.asarray(self.offsets, dtype=np.int64),
            labels=np.asarray(self.labels, dtype=np.int8),
            subtypes=np.asarray(self.subtypes, dtype=object),
        )


def array_nbytes(array: np.ndarray) -> int:
    total = int(array.nbytes)
    if array.dtype == object:
        total += sum(len(str(value).encode("utf-8")) for value in array)
    return total


def ragged_kmv_info(data: RaggedKMVDataset) -> Dict[str, Any]:
    representation_nbytes = int(data.hashes.nbytes + data.offsets.nbytes)
    dataset_nbytes = (
        representation_nbytes
        + array_nbytes(data.labels)
        + array_nbytes(data.subtypes)
    )
    lengths = np.diff(data.offsets)
    return {
        "shape": str((len(data), "ragged")),
        "hashes": int(len(data.hashes)),
        "representation_kb": float(representation_nbytes / 1024.0),
        "dataset_kb": float(dataset_nbytes / 1024.0),
        "dtype": str(data.hashes.dtype),
        "empty": int(np.sum(lengths == 0)),
        "min_length": int(np.min(lengths)) if len(lengths) else 0,
        "median_length": float(np.median(lengths)) if len(lengths) else 0.0,
        "max_length": int(np.max(lengths)) if len(lengths) else 0,
    }


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
    files = sorted(
        f for f in glob.glob(str(pattern), recursive=True)
        if is_real_parquet_path(f)
    )

    return files


def query_dir(query: str) -> Path:
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

    dir_names = [template.format(q=q) for template in ANOMALY_DIR_NAMES]
    candidates: List[Path] = []

    for dir_name in dir_names:
        candidates.append(query_dir(query) / dir_name / "**" / "*.parquet")
    for dir_name in dir_names:
        candidates.append(EXTERNAL_ROOT / dir_name / "**" / "*.parquet")

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


def stable_hash_u64(text: str) -> int:
    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
        key=b"prov-split-v1",
    ).digest()

    return int.from_bytes(digest, byteorder="big", signed=False)


def metadata_columns() -> List[str]:
    return ["query_id", "query_name", "subtype"]


def kmv_columns_for_method(method: str, kmv_k: int) -> List[str]:
    try:
        return METHOD_KMV_COLS[method][kmv_k]
    except KeyError:
        raise ValueError(f"Método/KMV inválido: method={method}, kmv_k={kmv_k}")


def find_kmv_column(schema_names: List[str], method: str, kmv_k: int, path: str) -> str:
    candidates = kmv_columns_for_method(method, kmv_k)

    for col in candidates:
        if col in schema_names:
            return col

    raise ValueError(
        f"Coluna KMV-{kmv_k} não encontrada em {path}. "
        f"Tentei: {candidates}"
    )


def data_columns_for_kmv(method: str, kmv_k: int, kmv_col: str) -> List[str]:
    return ["query_id", "query_name", "subtype", kmv_col]


def _as_python_list(values: Any) -> List[Any]:
    if values is None:
        return []

    if hasattr(values, "as_py"):
        values = values.as_py()

    if values is None:
        return []

    if isinstance(values, np.ndarray):
        return values.tolist()

    if isinstance(values, (list, tuple, set, frozenset)):
        return list(values)

    if isinstance(values, str):
        s = values.strip()

        if not s:
            return []

        if s.startswith("[") or s.startswith("(") or s.startswith("{"):
            try:
                parsed = ast.literal_eval(s)

                if isinstance(parsed, (list, tuple, set, frozenset)):
                    return list(parsed)

                return [parsed]

            except Exception:
                pass

        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]

        return [s]

    return [values]


def _signed_to_unsigned_u64(v: int) -> int:
    if v < 0:
        return v + (1 << 64)

    return v


def _to_ordered_unique_uint64_array(x: Any, kmv_k: int) -> np.ndarray:
    raw = _as_python_list(x)

    if not raw:
        return np.empty(0, dtype=np.uint64)

    vals: List[int] = []

    for v in raw[:kmv_k]:
        if v is None:
            continue

        vals.append(_signed_to_unsigned_u64(int(v)))

    if not vals:
        return np.empty(0, dtype=np.uint64)

    # Defensivo: garante ordem crescente e remove duplicados.
    vals = sorted(set(vals))[:kmv_k]

    return np.asarray(vals, dtype=np.uint64)


def load_metadata(pattern: str) -> List[MetaRow]:
    files = parquet_files(pattern)

    if not files:
        raise FileNotFoundError(f"Nenhum ficheiro encontrado: {pattern}")

    cols = metadata_columns()
    out: List[MetaRow] = []
    rows_seen = 0
    t0 = time.perf_counter()

    for file_i, path in enumerate(files, 1):
        last_error: Optional[OSError] = None
        for attempt in range(1, PARQUET_READ_ATTEMPTS + 1):
            pending: List[MetaRow] = []
            use_threads = PARQUET_USE_THREADS if attempt == 1 else False
            try:
                parquet_file = pq.ParquetFile(path)
                schema_names = set(parquet_file.schema_arrow.names)
                missing_required = {"query_id", "query_name"}.difference(
                    schema_names
                )
                if missing_required:
                    raise ValueError(
                        f"{path}: faltam colunas de identidade "
                        f"{sorted(missing_required)}"
                    )
                use_cols = [column for column in cols if column in schema_names]
                for batch in parquet_file.iter_batches(
                    columns=use_cols,
                    batch_size=PARQUET_BATCH_SIZE,
                    use_threads=use_threads,
                ):
                    rows = batch.to_pylist()
                    pending.extend(
                        MetaRow(
                            identity=make_identity(row),
                            subtype=str(row.get("subtype", "")),
                        )
                        for row in rows
                    )
                    del rows, batch
                out.extend(pending)
                rows_seen += len(pending)
                break
            except OSError as exc:
                last_error = exc
                print(
                    f"[PARQUET READ RETRY] stage=metadata "
                    f"attempt={attempt}/{PARQUET_READ_ATTEMPTS} "
                    f"threads={use_threads} path={path} error={exc}",
                    flush=True,
                )
                if attempt < PARQUET_READ_ATTEMPTS:
                    gc.collect()
                    time.sleep(PARQUET_RETRY_DELAY_S * attempt)
        else:
            assert last_error is not None
            raise OSError(
                getattr(last_error, "errno", 5),
                f"Falha persistente ao ler metadata do Parquet: {path}. "
                f"Tentativas={PARQUET_READ_ATTEMPTS}. "
                f"Erro original: {last_error}",
            ) from last_error

        print(f"  [META] {file_i}/{len(files)} rows={rows_seen:,}", flush=True)
        gc.collect()

    print(
        f"[META DONE] rows={len(out):,} time={time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    return out


SPLIT_NAMES = (
    "train_normal",
    "fit",
    "calib",
    "test_normal",
    "test_anom",
)


def split_manifest_path(query: str, scale_name: str) -> Path:
    return SPLIT_DIR / (
        f"{query.lower()}_{scale_name}_{SPLIT_FILE_SUFFIX}"
    )


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
            positioned[split_name].append(
                (int(row["pos"]), str(row["identity"]))
            )
    manifest: Dict[str, List[str]] = {}
    for split_name, rows in positioned.items():
        rows.sort(key=lambda item: item[0])
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
        raise ValueError("train_normal tem de ser a união exacta de fit e calib.")
    if not (train_normal | test_normal) <= normal_pool:
        missing = (train_normal | test_normal).difference(normal_pool)
        raise ValueError(f"Execuções normais ausentes: {sorted(missing)[:5]}")
    if not test_anom <= anomaly_pool:
        missing = test_anom.difference(anomaly_pool)
        raise ValueError(f"Anomalias ausentes: {sorted(missing)[:5]}")
    expected_counts = {
        "train_normal": train_k,
        "fit": train_k - max(1, int(train_k * CALIB_FRAC)),
        "calib": max(1, int(train_k * CALIB_FRAC)),
        "test_normal": test_normal_k,
        "test_anom": n_anom_test,
    }
    for split_name, expected in expected_counts.items():
        actual = len(manifest[split_name])
        if actual != expected:
            raise ValueError(
                f"Split {split_name}: manifesto={actual:,}, esperado={expected:,}"
            )


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
                f"[PARQUET READ RETRY] stage=identity "
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
        f"Falha persistente ao ler identidades do Parquet: {path}. "
        f"Tentativas={PARQUET_READ_ATTEMPTS}. Erro original: {last_error}",
    ) from last_error


def read_selected_kmv_rows_from_file(
    path: str,
    method: str,
    kmv_k: int,
    file_targets: set[str],
) -> List[Tuple[str, np.ndarray, str]]:
    last_error: Optional[OSError] = None
    for attempt in range(1, PARQUET_READ_ATTEMPTS + 1):
        use_threads = PARQUET_USE_THREADS if attempt == 1 else False
        pending_rows: List[Tuple[str, np.ndarray, str]] = []
        pending_identities: set[str] = set()
        try:
            parquet_file = pq.ParquetFile(path)
            schema_names = list(parquet_file.schema_arrow.names)
            kmv_col = find_kmv_column(
                schema_names=schema_names,
                method=method,
                kmv_k=kmv_k,
                path=path,
            )
            required = {"query_id", "query_name", kmv_col}
            missing = required.difference(schema_names)
            if missing:
                raise ValueError(
                    f"{path}: faltam colunas obrigatórias {sorted(missing)}"
                )
            use_cols = [
                column
                for column in data_columns_for_kmv(method, kmv_k, kmv_col)
                if column in schema_names
            ]
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
                            _to_ordered_unique_uint64_array(
                                row.get(kmv_col), kmv_k=kmv_k
                            ),
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
                    f"{len(missing_identities):,} identidades; "
                    f"exemplos={sorted(missing_identities)[:5]}"
                )
            return pending_rows
        except OSError as exc:
            last_error = exc
            pending_rows.clear()
            pending_identities.clear()
            print(
                f"[PARQUET READ RETRY] stage=payload kmv={kmv_k} "
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
        f"Falha persistente ao ler KMV-{kmv_k} do Parquet: {path}. "
        f"Tentativas={PARQUET_READ_ATTEMPTS}. Erro original: {last_error}",
    ) from last_error


def load_selected_kmv_dataset(
    pattern: str,
    method: str,
    kmv_k: int,
    label: int,
    selected_identities: set[str],
) -> RaggedKMVDataset:
    files = parquet_files(pattern)

    if not files:
        raise FileNotFoundError(f"Nenhum ficheiro encontrado: {pattern}")

    builder = RaggedKMVBuilder()
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
            pending_rows = read_selected_kmv_rows_from_file(
                path=path,
                method=method,
                kmv_k=kmv_k,
                file_targets=file_targets,
            )
            for identity, hashes, subtype in pending_rows:
                builder.append(
                    hashes=hashes,
                    label=label,
                    subtype=subtype,
                )
                loaded_identities.add(identity)
                rows_loaded += 1
            del pending_rows
        print(
            f"  [DATA] kmv={kmv_k} label={label} "
            f"{file_i}/{len(files)} rows_seen={rows_seen:,} "
            f"rows_loaded={rows_loaded:,}/{target:,}",
            flush=True,
        )
        gc.collect()
    data = builder.finalize()
    missing_selected = selected_identities.difference(loaded_identities)
    if len(data) != target or missing_selected:
        raise ValueError(
            f"Carregamento incompleto para {method}/KMV-{kmv_k}: "
            f"loaded={len(data):,} target={target:,}; "
            f"missing={len(missing_selected):,}; "
            f"examples={sorted(missing_selected)[:5]}"
        )
    info = ragged_kmv_info(data)
    print(
        f"[DATA DONE] kmv={kmv_k} label={label} rows={len(data):,} "
        f"hashes={len(data.hashes):,} empty={info['empty']:,} "
        f"min_len={info['min_length']} "
        f"median_len={info['median_length']:.1f} "
        f"max_len={info['max_length']} "
        f"time={time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return data


@njit(fastmath=True, cache=True)
def c_kmv_jaccard_sorted(
    a: np.ndarray,
    a_start: int,
    a_end: int,
    b: np.ndarray,
    b_start: int,
    b_end: int,
    kmv_k: int,
) -> float:
    n_a = a_end - a_start
    n_b = b_end - b_start

    if n_a == 0 and n_b == 0:
        return 1.0

    if n_a == 0 or n_b == 0:
        return 0.0

    # Interseção das assinaturas.
    i = a_start
    j = b_start
    inter = 0

    while i < a_end and j < b_end:
        av = a[i]
        bv = b[j]

        if av == bv:
            inter += 1
            i += 1
            j += 1
        elif av < bv:
            i += 1
        else:
            j += 1

    # Se ambas as assinaturas têm menos de k, o conjunto é completo.
    if n_a < kmv_k and n_b < kmv_k:
        union_size = n_a + n_b - inter

        if union_size == 0:
            return 1.0

        return inter / union_size

    # Estimador bottom-k coordenado:
    # J ≈ |intersection ∩ bottom_k(union)| / k
    i = a_start
    j = b_start

    union_seen = 0
    shared_in_bottom = 0

    while (i < a_end or j < b_end) and union_seen < kmv_k:
        if i < a_end and j < b_end:
            av = a[i]
            bv = b[j]

            if av == bv:
                union_seen += 1
                shared_in_bottom += 1
                i += 1
                j += 1
            elif av < bv:
                union_seen += 1
                i += 1
            else:
                union_seen += 1
                j += 1

        elif i < a_end:
            union_seen += 1
            i += 1

        else:
            union_seen += 1
            j += 1

    if union_seen == 0:
        return 1.0

    # Se a união das assinaturas observadas tiver menos de k, usa denominador real.
    if union_seen < kmv_k:
        return shared_in_bottom / union_seen

    return shared_in_bottom / kmv_k


@njit(fastmath=True, cache=True)
def c_insert_topk(top_scores: np.ndarray, score: float) -> None:
    min_pos = 0
    min_val = top_scores[0]

    for i in range(1, len(top_scores)):
        if top_scores[i] < min_val:
            min_val = top_scores[i]
            min_pos = i

    if score > min_val:
        top_scores[min_pos] = score


@njit(parallel=True, fastmath=True, cache=True)
def score_kmv_exact_all_fit_numba(
    eval_hashes: np.ndarray,
    eval_offsets: np.ndarray,
    fit_hashes: np.ndarray,
    fit_offsets: np.ndarray,
    k_neighbors: int,
    empty_score: float,
    kmv_k: int,
) -> np.ndarray:
    n_eval = len(eval_offsets) - 1
    n_fit = len(fit_offsets) - 1

    scores = np.zeros(n_eval, dtype=np.float32)

    k = k_neighbors

    if k < 1:
        k = 1

    if k > n_fit:
        k = n_fit

    for i in prange(n_eval):
        q_start = eval_offsets[i]
        q_end = eval_offsets[i + 1]

        if q_start == q_end:
            scores[i] = empty_score
            continue

        top_scores = np.full(k, -1.0, dtype=np.float32)

        for j in range(n_fit):
            t_start = fit_offsets[j]
            t_end = fit_offsets[j + 1]

            sim = c_kmv_jaccard_sorted(
                eval_hashes,
                q_start,
                q_end,
                fit_hashes,
                t_start,
                t_end,
                kmv_k,
            )

            c_insert_topk(top_scores, sim)

        mean_topk = 0.0

        for kk in range(k):
            if top_scores[kk] > 0.0:
                mean_topk += top_scores[kk]

        mean_topk /= k

        scores[i] = 1.0 - mean_topk

    return scores


def score_kmv_exact_all_fit(
    fit: RaggedKMVDataset,
    data: RaggedKMVDataset,
    kmv_k: int,
) -> np.ndarray:
    return score_kmv_exact_all_fit_numba(
        data.hashes,
        data.offsets,
        fit.hashes,
        fit.offsets,
        K_NEIGHBORS,
        EMPTY_PROV_SCORE,
        kmv_k,
    ).astype(np.float32, copy=False)


def compute_threshold_bootstrap(
    calib_scores: np.ndarray,
    percentile: int,
    B: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    thresholds = np.zeros(B, dtype=np.float64)

    for b in range(B):
        sample = rng.choice(
            calib_scores,
            size=len(calib_scores),
            replace=True,
        )

        thresholds[b] = np.percentile(sample, percentile)

    return (
        float(np.mean(thresholds)),
        float(np.percentile(thresholds, 2.5)),
        float(np.percentile(thresholds, 97.5)),
    )


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


def eval_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
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


def write_result_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted(row.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def run_one(
    query: str,
    scale: dict,
    method: str,
    kmv_k: int,
) -> Dict[str, Any]:
    scale_name = str(scale["name"])
    train_k = int(scale["train_k"])
    test_normal_k = int(scale["test_normal_k"])
    n_anom_test = int(scale["n_anom_test"])

    n_glob = normal_glob(query)
    a_glob = anomaly_glob(query)

    print("\n" + "#" * 100, flush=True)
    print(
        f"[QUERY] {query.upper()} | [ANOMALY] {ANOMALY_CLASS} | "
        f"[SCALE] {scale_name} | [METHOD] {method} | [KMV] {kmv_k}",
        flush=True,
    )
    print("#" * 100, flush=True)
    print(f"[PATH] normals={n_glob}", flush=True)
    print(f"[PATH] anomalies={a_glob}", flush=True)
    print(
        "[MODE] kmv_exact_all_fit | no candidates | "
        "no inverted index | two-pass loading",
        flush=True,
    )
    print(
        f"[SPLIT POLICY] class={ANOMALY_CLASS} | {SPLIT_POLICY} | "
        f"create_if_missing={ALLOW_CREATE_SHARED_SPLITS}",
        flush=True,
    )
    print(f"[SAFE] parquet_use_threads={PARQUET_USE_THREADS}", flush=True)

    clear_dmesg_best_effort()
    total_start = time.perf_counter()

    print("[PASS 1] load normal metadata", flush=True)
    metadata_start = time.perf_counter()
    normal_meta = load_metadata(n_glob)
    print("[PASS 1] load anomaly metadata", flush=True)
    anomaly_meta = load_metadata(a_glob)
    metadata_load_s = time.perf_counter() - metadata_start

    manifest_start = time.perf_counter()
    manifest, manifest_path = load_shared_split_manifest(
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
        f"[SPLIT NORMAL] train={n_train_original:,} "
        f"fit={len(fit_ids):,} calib={len(calib_ids):,} "
        f"test_normal={len(test_normal_ids):,}",
        flush=True,
    )
    print(f"[SPLIT ANOM] test_anom={len(test_anom_ids):,}", flush=True)

    print("[PASS 2] load fit KMV only", flush=True)
    load_fit_start = time.perf_counter()
    fit = load_selected_kmv_dataset(
        n_glob,
        method,
        kmv_k,
        label=0,
        selected_identities=fit_ids,
    )
    load_fit_s = time.perf_counter() - load_fit_start
    fit_info = ragged_kmv_info(fit)
    del fit_ids
    gc.collect()
    print(f"[FIT] rows={len(fit):,} hashes={len(fit.hashes):,}", flush=True)

    print("[PASS 2] load calib KMV only", flush=True)
    load_calib_start = time.perf_counter()
    calib = load_selected_kmv_dataset(
        n_glob,
        method,
        kmv_k,
        label=0,
        selected_identities=calib_ids,
    )
    load_calib_s = time.perf_counter() - load_calib_start
    calib_info = ragged_kmv_info(calib)
    del calib_ids
    gc.collect()

    print("[SCORE] calib KMV exact all-fit", flush=True)
    t_score_calib = time.perf_counter()
    calib_scores = score_kmv_exact_all_fit(
        fit=fit,
        data=calib,
        kmv_k=kmv_k,
    )
    score_calib_s = time.perf_counter() - t_score_calib
    print(
        f"[SCORE DONE] calib rows={len(calib):,} time={score_calib_s:.2f}s",
        flush=True,
    )
    n_calib = len(calib)
    del calib
    gc.collect()

    print("[PASS 2] load test_normal KMV only", flush=True)
    load_normal_start = time.perf_counter()
    test_normal = load_selected_kmv_dataset(
        n_glob,
        method,
        kmv_k,
        label=0,
        selected_identities=test_normal_ids,
    )
    load_normal_s = time.perf_counter() - load_normal_start
    test_normal_info = ragged_kmv_info(test_normal)
    del test_normal_ids
    gc.collect()

    print("[SCORE] test_normal KMV exact all-fit", flush=True)
    t_score_normal = time.perf_counter()
    normal_scores = score_kmv_exact_all_fit(
        fit=fit,
        data=test_normal,
        kmv_k=kmv_k,
    )
    score_normal_s = time.perf_counter() - t_score_normal
    print(
        f"[SCORE DONE] test_normal rows={len(test_normal):,} time={score_normal_s:.2f}s",
        flush=True,
    )
    n_test_normal = len(test_normal)
    del test_normal
    gc.collect()

    print("[PASS 2] load test_anom KMV only", flush=True)
    load_anom_start = time.perf_counter()
    test_anom = load_selected_kmv_dataset(
        a_glob,
        method,
        kmv_k,
        label=1,
        selected_identities=test_anom_ids,
    )
    load_anom_s = time.perf_counter() - load_anom_start
    test_anom_info = ragged_kmv_info(test_anom)
    del test_anom_ids
    gc.collect()

    print("[SCORE] test_anom KMV exact all-fit", flush=True)
    t_score_anom = time.perf_counter()
    anom_scores = score_kmv_exact_all_fit(
        fit=fit,
        data=test_anom,
        kmv_k=kmv_k,
    )
    score_anom_s = time.perf_counter() - t_score_anom
    print(
        f"[SCORE DONE] test_anom rows={len(test_anom):,} time={score_anom_s:.2f}s",
        flush=True,
    )
    n_anom = len(test_anom)
    n_fit = len(fit)
    del test_anom, fit
    gc.collect()

    test_scores = np.concatenate(
        [normal_scores, anom_scores],
    ).astype(np.float32, copy=False)

    y_true = np.concatenate(
        [
            np.zeros(len(normal_scores), dtype=np.int8),
            np.ones(len(anom_scores), dtype=np.int8),
        ]
    )

    metrics_start = time.perf_counter()
    base = eval_scores(
        y_true=y_true,
        scores=test_scores,
        threshold=np.inf,
    )
    contamination = float(n_anom / max(1, len(test_scores)))
    load_s = (
        metadata_load_s
        + load_fit_s
        + load_calib_s
        + load_normal_s
        + load_anom_s
    )
    score_total_s = score_calib_s + score_normal_s + score_anom_s
    representation_total_kb = (
        fit_info["representation_kb"]
        + calib_info["representation_kb"]
        + test_normal_info["representation_kb"]
        + test_anom_info["representation_kb"]
    )

    result: Dict[str, Any] = {
        "query": query,
        "anomaly_class": ANOMALY_CLASS,
        "method": method,
        "scale": scale_name,
        "code_version": CODE_VERSION,
        "code_sha256": script_sha256(),
        "split_policy": SPLIT_POLICY,
        "representation": (
            f"lineage_kmv_{kmv_k}"
            if method == "lineage_jaccard"
            else f"atomic_why_kmv_{kmv_k}"
        ),
        "representation_family": "kmv_ragged_set",
        "representation_dim": kmv_k,
        "kmv_k": kmv_k,
        "sketch_budget_bytes_per_execution": kmv_k * 8,
        "candidate_policy": "exact_all_fit_no_candidates",
        "train_k": train_k,
        "test_normal_k": test_normal_k,
        "n_anom_test_config": n_anom_test,
        "n_train": n_train_original,
        "n_fit": n_fit,
        "n_fit_actual": n_fit,
        "n_calib": n_calib,
        "n_test": len(test_scores),
        "n_test_normal": n_test_normal,
        "n_anom": n_anom,
        "n_test_anom": n_anom,
        "contamination": contamination,
        "k_neighbors": K_NEIGHBORS,
        "empty_prov_score": EMPTY_PROV_SCORE,
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
        "load_protocol": "metadata_pass_then_selected_ragged_kmv_loading",
        "quality_protocol": "shared_split_threshold_from_calib_percentile",
        "cost_protocol": "parquet_load_plus_stored_kmv_set_1nn",
        "runtime_comparison_mode": "stored_kmv_ragged_set_1nn",
        "comparison_ready": True,
        "xfit_shape": fit_info["shape"],
        "xfit_hashes": fit_info["hashes"],
        "xfit_kb": fit_info["representation_kb"],
        "xfit_dataset_kb": fit_info["dataset_kb"],
        "xfit_dtype": fit_info["dtype"],
        "xcalib_shape": calib_info["shape"],
        "xcalib_hashes": calib_info["hashes"],
        "xcalib_kb": calib_info["representation_kb"],
        "xcalib_dataset_kb": calib_info["dataset_kb"],
        "xtest_normal_shape": test_normal_info["shape"],
        "xtest_normal_hashes": test_normal_info["hashes"],
        "xtest_normal_kb": test_normal_info["representation_kb"],
        "xtest_normal_dataset_kb": test_normal_info["dataset_kb"],
        "xtest_anom_shape": test_anom_info["shape"],
        "xtest_anom_hashes": test_anom_info["hashes"],
        "xtest_anom_kb": test_anom_info["representation_kb"],
        "xtest_anom_dataset_kb": test_anom_info["dataset_kb"],
        "xtest_shape": str((n_test_normal + n_anom, "ragged")),
        "xtest_kb": (
            test_normal_info["representation_kb"]
            + test_anom_info["representation_kb"]
        ),
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
        "representation_build_s": 0.0,
        "fit_s": 0.0,
        "preprocess_s": load_s,
        "preprocess_total_s": load_s,
    }

    for pct in THRESHOLD_PERCENTILES:
        thr_mean, ci_lo, ci_hi = compute_threshold_bootstrap(
            calib_scores,
            pct,
            BOOTSTRAP_B,
            RANDOM_SEED,
        )

        metrics = eval_scores(
            y_true=y_true,
            scores=test_scores,
            threshold=thr_mean,
        )

        suffix = f"_pct{pct}"

        result[f"threshold{suffix}"] = metrics["threshold"]
        result[f"thr_ci_lo{suffix}"] = ci_lo
        result[f"thr_ci_hi{suffix}"] = ci_hi
        result[f"f1{suffix}"] = metrics["f1"]
        result[f"precision{suffix}"] = metrics["precision"]
        result[f"recall{suffix}"] = metrics["recall"]
        result[f"n_pred{suffix}"] = metrics["n_pred"]

        print(
            f"    pct{pct}: "
            f"thr={thr_mean:.6f} "
            f"f1={metrics['f1']:.4f} "
            f"p={metrics['precision']:.4f} "
            f"r={metrics['recall']:.4f} "
            f"n_pred={metrics['n_pred']}",
            flush=True,
        )

    result["metrics_s"] = float(time.perf_counter() - metrics_start)
    result["total_s"] = float(load_s + score_total_s)
    result["end_to_end_s"] = float(time.perf_counter() - total_start)
    result["runtime_s"] = result["total_s"]
    result["peak_rss_mb"] = float(peak_rss_mb())
    result["status"] = "ok"
    result["error"] = ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    score_path = SCORE_DIR / (
        f"{query}{FILE_TAG}_{scale_name}_{method}_kmv{kmv_k}_scores.npz"
    )
    np.savez_compressed(
        score_path,
        calibration=calib_scores,
        test_normal=normal_scores,
        test_anomaly=anom_scores,
    )
    result["scores_file"] = str(score_path)

    if ANOMALY_CLASS == "S1":
        out_csv = OUT_DIR / (
            f"{query}_set_kmv{kmv_k}_exact_twopass_{method}_{scale_name}.csv"
        )
    else:
        out_csv = OUT_DIR / (
            f"{query}_s2_set_kmv{kmv_k}_exact_twopass_{method}_{scale_name}.csv"
        )
    write_result_row(out_csv, result)
    print(
        f"[OK] guardado -> {out_csv} "
        f"ap={result['ap']:.4f} "
        f"auc={result['auc']:.4f} "
        f"time={result['runtime_s']:.2f}s",
        flush=True,
    )

    check_dmesg_best_effort(f"{query}/{ANOMALY_CLASS.lower()}/{scale_name}/{method}/kmv{kmv_k}")
    del calib_scores, normal_scores, anom_scores, test_scores, y_true
    gc.collect()
    return result


def warmup_numba() -> None:
    print("[INIT] warm-up Numba", flush=True)

    hashes = np.asarray([1, 2, 3], dtype=np.uint64)
    offsets = np.asarray([0, 3], dtype=np.int64)

    _ = score_kmv_exact_all_fit_numba(
        hashes,
        offsets,
        hashes,
        offsets,
        K_NEIGHBORS,
        EMPTY_PROV_SCORE,
        512,
    )


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



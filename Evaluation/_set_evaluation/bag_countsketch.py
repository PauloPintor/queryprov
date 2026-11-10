# Bag CountSketch only; original sidecar contracts retained.
from __future__ import annotations


import ast


import contextlib


import csv


import fcntl


import hashlib


import io


import json


import math


import multiprocessing


import os


import platform


import re


import resource


import sys


import tempfile


import time


import traceback


from collections import Counter


from pathlib import Path


import numpy as np


import pyarrow as pa


import pyarrow.parquet as pq


import sklearn


import scipy


from sklearn.metrics import average_precision_score, roc_auc_score


from threadpoolctl import threadpool_info, threadpool_limits


ANOMALY_CLASSES = ("S1", "S2")


QUERY_IDS = (3, 5, 18, 20, 28, 60, 66, 82, 84, 85, 89, 92, 93, 94, 98)


FEATURES = ("bag_lineage", "bag_atomic_why")


SKETCH_DIMS = (512, 1024)


SCALES = (
    {"name": "5k", "train_k": 5000, "test_normal_k": 1250, "n_anom_test": 66},
    {"name": "10k", "train_k": 10000, "test_normal_k": 2500, "n_anom_test": 125},
    {"name": "25k", "train_k": 25000, "test_normal_k": 6250, "n_anom_test": 313},
    {"name": "50k", "train_k": 50000, "test_normal_k": 12500, "n_anom_test": 625},
)


CALIB_FRAC = 0.20


BOOTSTRAP_B = 1000


RANDOM_SEED = 13


THRESHOLD_PERCENTILES = (95, 99)


BLAS_THREADS = 4


SCORE_BATCH_ROWS = 256


PARQUET_BATCH_ROWS = 256


MAX_MATRIX_GIB = 4.0


WORKER_TIMEOUT_SECONDS = 5 * 60 * 60


CODE_VERSION = "2026-11-10-bag-countsketch-s1-s2-knn-cli"


BAG_PSD_VERSION = "2026-10-01-weighted-cs-psd-v1"


LEGACY_ID_OFFSETS = {(q, "s1"): 65000 for q in
                     (3, 5, 18, 20, 60, 66, 82, 84, 85, 92, 93, 94, 98)}


LEGACY_ID_OFFSETS.update({(5, "s2"): 66050, (84, "s2"): 66050})


SPLITS = ("train_normal", "fit", "calib", "test_normal", "test_anom")


MATRIX_SPLITS = SPLITS[1:]


ALGORITHMS = {"bag_lineage": "prov-cs-v2-lineage",
              "bag_atomic_why": "atomic-witness-countsketch-splitmix64-v1"}


IDENTITY_COLUMNS = ("query_id", "template_id", "query_name", "subtype", "sql_sha256")


BASELINE_COLUMNS = tuple(f"{family}_{dim}" for family in (
    "lineage_kmv", "monoids_kmv", "lineage_countsketch", "atomic_countsketch")
    for dim in (512, 1024))


CONTRACT_COLUMNS = (*IDENTITY_COLUMNS, "query", "pool", "bag_psd_version",
    "bag_weights", "witness_semantics", "bag_vectors_normalised", "vector_sketch_seed",
    "lineage_sketch_algorithm", "atomic_sketch_algorithm", "bag_lineage_population",
    "bag_atomic_population", "factor_sample_r", "factor_sizes",
    "bag_atomic_selected_unique_decimal", "bag_atomic_selected_total_decimal",
    "nwitness_unique_decimal", "nwitness_total_decimal", "baseline_columns_verified")


MAIN_CONTRACT_COLUMNS = (*IDENTITY_COLUMNS, "query", "pool", "vector_sketch_weights",
    "vector_sketch_normalised", "vector_sketch_seed", "lineage_exact", "atomic_exact",
    "lineage_sketch_algorithm", "atomic_sketch_algorithm", "atomic_witness_hash_namespace",
    "provenance_mode", "factor_sample_r", "factor_sample_seed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_object(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def computation_fingerprint() -> str:
    nodes = ast.parse(Path(__file__).read_text(encoding="utf-8")).body
    definitions = [ast.dump(n, include_attributes=False) for n in nodes
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    return digest_object({"version": CODE_VERSION, "definitions": definitions,
                          "algorithms": ALGORITHMS, "contract": CONTRACT_COLUMNS,
                          "numpy": np.__version__, "sklearn": sklearn.__version__,
                          "pyarrow": pa.__version__, "python": platform.python_version(),
                          "scipy": scipy.__version__, "bag_version": BAG_PSD_VERSION,
                          "main_contract": MAIN_CONTRACT_COLUMNS})


def vector_column(feature: str, dim: int) -> str:
    family = {"bag_lineage": "lineage", "bag_atomic_why": "atomic"}[feature]
    return f"bag_{family}_countsketch_{dim}"


def split_path(anomaly_class: str, query: int, scale: str) -> Path:
    tag = "_s2" if anomaly_class == "S2" else ""
    return split_directory(anomaly_class) / f"q{query}_{scale}{tag}_shared_split_ids.csv"


def split_directory(anomaly_class: str) -> Path:
    directory = Path(SPLIT_DIRS[anomaly_class])
    if directory.is_dir():
        return directory
    if anomaly_class == "S2" and directory.name.lower() == "splits_s2" and directory.parent.is_dir():
        matches = [p for p in directory.parent.iterdir()
                   if p.is_dir() and p.name.lower() == "splits_s2"]
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(f"Existing KMV split folder missing: {directory}. It will not be created.")


def load_manifest(path: Path, query: int, scale: dict, calib_frac: float) -> dict:
    # Read once, so the recorded hash is the exact manifest parsed here.
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    required = {"query", "scale", "split", "pos", "identity", "subtype", "label"}
    if not required <= set(reader.fieldnames or ()):
        raise ValueError(f"{path}: missing split columns {sorted(required - set(reader.fieldnames or ())) }")
    grouped = {name: [] for name in SPLITS}
    for row in reader:
        name = row["split"]
        if row["query"].lower() != f"q{query}" or row["scale"] != scale["name"] or name not in grouped:
            raise ValueError(f"{path}: wrong query, scale or split: {row}")
        identity = row["identity"]
        if not re.fullmatch(r"[0-9]+\|[^|/\\]+", identity):
            raise ValueError(f"{path}: invalid identity {identity!r}")
        if int(row["label"]) != int(name == "test_anom"):
            raise ValueError(f"{path}: wrong label for {name}")
        grouped[name].append((int(row["pos"]), identity, row["subtype"]))
    n_calib = max(1, int(scale["train_k"] * calib_frac))
    expected = {"train_normal": scale["train_k"], "fit": scale["train_k"] - n_calib,
                "calib": n_calib, "test_normal": scale["test_normal_k"],
                "test_anom": scale["n_anom_test"]}
    result = {}
    for name, rows in grouped.items():
        rows.sort()
        ids = [row[1] for row in rows]
        if ([row[0] for row in rows] != list(range(len(rows))) or len(set(ids)) != len(ids)
                or len(rows) != expected[name]):
            raise ValueError(f"{path}: invalid {name} positions/identities/count; "
                             f"got {len(rows)}, expected {expected[name]}")
        result[name] = {"identities": ids, "subtypes": [row[2] for row in rows]}
    members = {name: set(result[name]["identities"]) for name in SPLITS}
    if members["train_normal"] != members["fit"] | members["calib"]:
        raise ValueError(f"{path}: train_normal differs from fit union calib")
    for i, left in enumerate(MATRIX_SPLITS):
        for right in MATRIX_SPLITS[i + 1:]:
            if members[left] & members[right]:
                raise ValueError(f"{path}: overlap between {left} and {right}")
    train_subtypes = dict(zip(result["train_normal"]["identities"], result["train_normal"]["subtypes"]))
    for name in ("fit", "calib"):
        if any(train_subtypes[i] != s for i, s in zip(result[name]["identities"], result[name]["subtypes"])):
            raise ValueError(f"{path}: subtype mismatch in {name}")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "splits": result}


def file_stamp(path: Path) -> list:
    info = path.stat()
    return [path.name, info.st_size, info.st_mtime_ns]


def snapshot_pool(directory: Path) -> dict:
    marker = directory / "_SUCCESS.json"
    bag_marker = directory / "_BAG_PSD_SUCCESS.json"
    if not marker.is_file() or not bag_marker.is_file():
        raise ValueError(f"Both _SUCCESS.json and _BAG_PSD_SUCCESS.json are required: {directory}")
    info = json.loads(marker.read_text())
    bags = json.loads(bag_marker.read_text())
    paths = sorted(p for p in directory.iterdir() if re.fullmatch(r"batch_[0-9]+\.parquet", p.name))
    if not paths or len(paths) != int(info["batches"]):
        raise ValueError(f"{directory}: main batch count disagrees with _SUCCESS.json")
    if (bags.get("bag_psd_version") != BAG_PSD_VERSION or bags.get("rows") != info["rows"]
            or not isinstance(bags.get("batches"), dict)
            or set(bags["batches"]) != {p.name for p in paths}
            or any(not isinstance(d, str) or not re.fullmatch(r"[0-9a-f]{64}", d)
                   for d in bags["batches"].values())):
        raise ValueError(f"{directory}: incomplete or incompatible bag success marker")
    sidecars = [p.with_name(p.stem + ".bagcompact.parquet") for p in paths]
    present = {p.name for p in directory.iterdir()
               if re.fullmatch(r"batch_[0-9]+\.bagcompact\.parquet", p.name)}
    if present != {p.name for p in sidecars} or any(not p.is_file() for p in sidecars):
        raise ValueError(f"{directory}: missing or orphaned bag sidecars")
    return {"directory": str(directory.resolve()), "files": [file_stamp(p) for p in sidecars],
            "main_files": [file_stamp(p) for p in paths], "marker_sha256": sha256_file(marker),
            "bag_marker_sha256": sha256_file(bag_marker), "bag_digests": bags["batches"],
            "bag_query": bags["query"], "bag_pool": bags["pool"], "rows": int(info["rows"])}


def validate_sidecar_binding(parquet: pq.ParquetFile, filename: str, snapshot: dict) -> None:
    main_name = filename.replace(".bagcompact.parquet", ".parquet")
    metadata = parquet.schema_arrow.metadata or {}
    if (metadata.get(b"bag_psd_input_sha256") != snapshot["bag_digests"][main_name].encode()
            or metadata.get(b"bag_psd_version") != BAG_PSD_VERSION.encode()):
        raise ValueError(f"Sidecar metadata does not match its completion marker: {filename}")


def validate_main_alignment(main: dict, bag: dict, query: int, pool: str) -> None:
    if any(main[k] != bag[k] for k in IDENTITY_COLUMNS):
        raise ValueError(f"Main/sidecar identity, order or SQL mismatch: {bag['query_name']}")
    if (str(main["query"]).lower() != f"q{query}" or main["pool"] != pool
            or main["lineage_exact"] is not True or main["vector_sketch_seed"] != 13
            or main["vector_sketch_weights"] != "set_membership"
            or main["vector_sketch_normalised"] is not False
            or main["lineage_sketch_algorithm"] != ALGORITHMS["bag_lineage"]
            or main["atomic_sketch_algorithm"] != ALGORITHMS["bag_atomic_why"]
            or main["atomic_witness_hash_namespace"] != "q28-vectorized-atomic-kmv-v1"):
        raise ValueError(f"Unexpected main Parquet contract: {bag['query_name']}")
    bounded = bag["bag_atomic_population"] == "bounded_factor_expansion"
    if bounded:
        if (main["atomic_exact"] is not False or main["provenance_mode"] != "bounded_factor_expansion"
                or main["factor_sample_r"] != bag["factor_sample_r"]
                or main["factor_sample_seed"] != "q28-factor-sample-v1"):
            raise ValueError(f"Main/sidecar bounded population differs: {bag['query_name']}")
    elif (main["atomic_exact"] is not True
          or main["provenance_mode"] not in {"exact_expansion", "exact_set_membership"}):
        raise ValueError(f"Main/sidecar full population differs: {bag['query_name']}")


def inspect_pool(query: int, pool: str) -> dict:
    suffix = {"normal": "normal", "s1": "anomalies", "s2": "s2_anomalies"}[pool]
    directory = INPUT_ROOT / f"Q{query}" / f"q{query}_{suffix}"
    snapshot = snapshot_pool(directory)
    if snapshot["bag_query"] != query or snapshot["bag_pool"] != pool:
        raise ValueError(f"Bag marker query/pool differs: {directory}")
    required = set(CONTRACT_COLUMNS) | {vector_column(f, d) for f in FEATURES for d in SKETCH_DIMS}
    rows = 0
    for index, (filename, _, _) in enumerate(snapshot["files"], 1):
        with pq.ParquetFile(directory / filename) as parquet:
            validate_sidecar_binding(parquet, filename, snapshot)
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{directory / filename}: missing bag/contract columns {sorted(missing)}")
            for feature in FEATURES:
                for dim in SKETCH_DIMS:
                    data_type = parquet.schema_arrow.field(vector_column(feature, dim)).type
                    if (not (pa.types.is_list(data_type) or pa.types.is_large_list(data_type))
                            or data_type.value_type != pa.float64()):
                        raise ValueError(f"Bag vector must preserve stored float64 values: {filename}")
            main_path = directory / filename.replace(".bagcompact.parquet", ".parquet")
            with pq.ParquetFile(main_path) as main_parquet:
                missing_main = set(MAIN_CONTRACT_COLUMNS) - set(main_parquet.schema_arrow.names)
                if missing_main or main_parquet.metadata.num_rows != parquet.metadata.num_rows:
                    raise ValueError(f"Main/sidecar schema or row-count mismatch: {main_path}")
                main_batches = main_parquet.iter_batches(columns=list(MAIN_CONTRACT_COLUMNS), batch_size=256)
                bag_batches = parquet.iter_batches(columns=list(CONTRACT_COLUMNS), batch_size=256)
                checked = 0
                for main_batch, bag_batch in zip(main_batches, bag_batches):
                    if main_batch.num_rows != bag_batch.num_rows:
                        raise ValueError(f"Main/sidecar batch alignment differs: {filename}")
                    for main_row, bag_row in zip(main_batch.to_pylist(), bag_batch.to_pylist()):
                        validate_contract(bag_row, query, pool)
                        validate_main_alignment(main_row, bag_row, query, pool)
                        checked += 1
                if checked != parquet.metadata.num_rows:
                    raise ValueError(f"Main/sidecar coverage differs: {filename}")
            rows += parquet.metadata.num_rows
        if index % 500 == 0:
            print(f"[PREFLIGHT] Q{query} {pool}: {index}/{len(snapshot['files'])} pairs", flush=True)
    if rows != snapshot["rows"]:
        raise ValueError(f"{directory}: row count disagrees with _SUCCESS.json")
    print(f"[INPUT OK] Q{query} {pool}: {rows:,} rows; {len(snapshot['files']):,} aligned bag sidecars", flush=True)
    return snapshot


def assert_inputs_unchanged(job: dict) -> None:
    if sha256_file(Path(job["manifest"]["path"])) != job["manifest"]["sha256"]:
        raise ValueError("Split manifest changed. No result may be reused or published.")
    for expected in job["pools"].values():
        if snapshot_pool(Path(expected["directory"])) != expected:
            raise ValueError(f"Compact inputs changed: {expected['directory']}")


def validate_contract(row: dict, query: int, pool: str) -> None:
    if str(row["query"]).lower() != f"q{query}" or row["pool"] != pool:
        raise ValueError(f"Wrong query/pool for {row['query_name']}")
    if (row["bag_weights"] != "witness_occurrence_frequency" or row["bag_vectors_normalised"] is not False
            or row["vector_sketch_seed"] != 13 or row["bag_psd_version"] != BAG_PSD_VERSION
            or row["witness_semantics"] != "set_of_distinct_tokens"):
        raise ValueError(f"Not a raw seed-13 frequency-weighted bag sketch: {row['query_name']}")
    if (row["lineage_sketch_algorithm"] != ALGORITHMS["bag_lineage"]
            or row["atomic_sketch_algorithm"] != ALGORITHMS["bag_atomic_why"]
            or row["bag_lineage_population"] != "full"
            or row["baseline_columns_verified"] is None
            or len(row["baseline_columns_verified"]) != len(BASELINE_COLUMNS)
            or set(row["baseline_columns_verified"]) != set(BASELINE_COLUMNS)):
        raise ValueError(f"Unexpected bag hash/baseline/population policy: {row['query_name']}")
    fields = ("nwitness_unique_decimal", "nwitness_total_decimal",
              "bag_atomic_selected_unique_decimal", "bag_atomic_selected_total_decimal")
    if any(not isinstance(row[k], str) or not re.fullmatch(r"[0-9]+", row[k]) for k in fields):
        raise ValueError(f"Invalid witness population counters: {row['query_name']}")
    full_unique, full_total, selected, selected_total = (int(row[k]) for k in fields)
    if not (0 <= selected <= full_unique <= full_total and selected <= selected_total <= full_total):
        raise ValueError(f"Invalid bag population counts: {row['query_name']}")
    if row["bag_atomic_population"] == "bounded_factor_expansion":
        if query != 28 or row["factor_sample_r"] != 14 or selected >= full_unique:
            raise ValueError(f"Unexpected bounded Atomic bag population: {row['query_name']}")
        factors = row["factor_sizes"]
        if (not factors or any(not isinstance(n, int) or n <= 0 for n in factors)
                or selected != math.prod(min(14, n) for n in factors)):
            raise ValueError(f"Selected factor population does not match stored factors: {row['query_name']}")
    elif row["bag_atomic_population"] != "full" or (selected, selected_total) != (full_unique, full_total):
        raise ValueError(f"Incomplete full Atomic bag population: {row['query_name']}")


def resolve_pool_identities(query: int, pool: str, snapshot: dict, manifests: list[dict]) -> dict:
    names = ("fit", "calib", "test_normal") if pool == "normal" else ("test_anom",)
    expected = {}
    for manifest in manifests:
        for split in names:
            selection = manifest["splits"][split]
            for identity, subtype in zip(selection["identities"], selection["subtypes"]):
                name = identity.split("|", 1)[1]
                previous = expected.setdefault(name, (identity, subtype))
                if previous != (identity, subtype):
                    raise ValueError(f"Conflicting split identity/subtype for Q{query} {pool}: {name}")
    directory = Path(snapshot["directory"])
    seen_names, seen_ids, found, offsets, translations = set(), set(), set(), set(), {}
    for filename, _, _ in snapshot["files"]:
        with pq.ParquetFile(directory / filename) as parquet:
            for batch in parquet.iter_batches(columns=list(CONTRACT_COLUMNS), batch_size=4096):
                for row in batch.to_pylist():
                    name, actual_id = str(row["query_name"]), int(row["query_id"])
                    if name in seen_names or actual_id in seen_ids:
                        raise ValueError(f"Duplicate selected identity or pool name/ID in {directory}: {actual_id}|{name}")
                    seen_names.add(name)
                    seen_ids.add(actual_id)
                    if name not in expected:
                        continue
                    wanted, subtype = expected[name]
                    actual = f"{actual_id}|{name}"
                    if str(row["subtype"]) != subtype:
                        raise ValueError(f"Subtype mismatch for {wanted}: split={subtype}, data={row['subtype']}")
                    validate_contract(row, query, pool)
                    sql_hash = row["sql_sha256"]
                    if not isinstance(sql_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", sql_hash):
                        raise ValueError(f"Invalid SQL provenance fingerprint for {wanted}")
                    offset = int(wanted.split("|", 1)[0]) - actual_id
                    offsets.add(offset)
                    if offset:
                        if offset != LEGACY_ID_OFFSETS.get((query, pool)) or actual_id < 0:
                            raise ValueError(f"Split identities missing: Q{query} {pool} needs {wanted}, "
                                             f"but data has {actual}; unrecognised ID offset {offset}")
                        sql_path = directory / (name + ".sql")
                        sql = sql_path.read_text(encoding="utf-8-sig")
                        if hashlib.sha256(sql.encode("utf-8")).hexdigest() != sql_hash:
                            raise ValueError(f"SQL hash mismatch for {sql_path}; ID translation refused")
                        translations[wanted] = {"parquet_identity": actual, "subtype": subtype,
                                                "sql_sha256": sql_hash}
                    found.add(name)
    missing = set(expected) - found
    if missing:
        raise ValueError(f"{directory}: {len(missing)} split identities missing; names={sorted(missing)[:5]}")
    if len(offsets) > 1:
        raise ValueError(f"Inconsistent numeric ID offsets in Q{query} {pool}: {sorted(offsets)}")
    mode = f"verified numeric ID offset={next(iter(offsets))}" if translations else "exact ID/name/subtype"
    print(f"[SPLIT OK] Q{query} {pool}: {len(found):,} selected executions; {mode}", flush=True)
    return translations


def parquet_identity(job: dict, pool: str, identity: str) -> str:
    return job.get("identity_translations", {}).get(pool, {}).get(identity, {}).get("parquet_identity", identity)


def load_matrices(job: dict) -> tuple[dict, dict]:
    splits = job["manifest"]["splits"]
    dim = job["dim"]
    config = job["config"]
    required_bytes = sum(len(splits[s]["identities"]) for s in MATRIX_SPLITS) * dim * 8 * 1.5
    if required_bytes > config["max_matrix_gib"] * 1024**3:
        raise MemoryError(f"Selected matrices need {required_bytes / 1024**3:.3f} GiB")
    matrices = {name: np.empty((len(splits[name]["identities"]), dim), dtype=np.float64)
                for name in MATRIX_SPLITS}
    sql_hashes = {name: [""] * len(matrix) for name, matrix in matrices.items()}
    bounded = Counter()
    column = vector_column(job["feature"], dim)
    for pool_name, snapshot in job["pools"].items():
        names = ("fit", "calib", "test_normal") if pool_name == "normal" else ("test_anom",)
        translations = job.get("identity_translations", {}).get(pool_name, {})
        destinations = {parquet_identity(job, pool_name, identity): (name, pos, splits[name]["subtypes"][pos])
                        for name in names for pos, identity in enumerate(splits[name]["identities"])}
        seen = set()
        directory = Path(snapshot["directory"])
        for file_index, (filename, _, _) in enumerate(snapshot["files"], 1):
            with pq.ParquetFile(directory / filename) as parquet:
                validate_sidecar_binding(parquet, filename, snapshot)
                for batch in parquet.iter_batches(batch_size=config["parquet_batch_rows"],
                                                   columns=[*CONTRACT_COLUMNS, column]):
                    # Convert only selected vector values, not every Arrow list in the batch.
                    metadata = batch.select(CONTRACT_COLUMNS).to_pylist()
                    vectors = batch.column(batch.schema.get_field_index(column))
                    for i, row in enumerate(metadata):
                        identity = f"{int(row['query_id'])}|{row['query_name']}"
                        if identity not in destinations:
                            continue
                        if identity in seen:
                            raise ValueError(f"Duplicate selected identity in {directory}: {identity}")
                        name, pos, subtype = destinations[identity]
                        if str(row["subtype"]) != subtype:
                            raise ValueError(f"Subtype mismatch for {identity}: split={subtype}, data={row['subtype']}")
                        validate_contract(row, job["query"], pool_name)
                        vector = np.asarray(vectors[i].as_py(), dtype=np.float64)
                        if vector.shape != (dim,) or not np.isfinite(vector).all():
                            raise ValueError(f"Invalid {column} for {identity}: {vector.shape}")
                        matrices[name][pos] = vector
                        sql_hash = row["sql_sha256"]
                        if not isinstance(sql_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", sql_hash):
                            raise ValueError(f"Invalid SQL provenance fingerprint for {identity}")
                        original = splits[name]["identities"][pos]
                        if original in translations and translations[original]["sql_sha256"] != sql_hash:
                            raise ValueError(f"SQL hash changed after identity preflight: {original}")
                        sql_hashes[name][pos] = sql_hash
                        bounded[name] += int(row["bag_atomic_population"] == "bounded_factor_expansion")
                        seen.add(identity)
            if file_index % 500 == 0:
                print(f"[LOAD] {pool_name} {file_index}/{len(snapshot['files'])}; selected={len(seen):,}", flush=True)
        missing = set(destinations) - seen
        if missing:
            raise ValueError(f"{directory}: {len(missing)} split identities missing; examples={sorted(missing)[:5]}. "
                             "Only preflight-verified ID translations are allowed; no resampling or split recreation.")
    audit = {f"{name}_atomic_bounded_rows": bounded[name] for name in MATRIX_SPLITS}
    audit.update({f"{name}_raw_vectors_sha256": hashlib.sha256(memoryview(matrix).cast("B")).hexdigest()
                  for name, matrix in matrices.items()})
    audit.update({f"{name}_selected_sql_sha256": digest_object(sql_hashes[name]) for name in MATRIX_SPLITS})
    audit["identity_translation_count"] = sum(len(t) for t in job.get("identity_translations", {}).values())
    audit["identity_translation_sha256"] = digest_object(job.get("identity_translations", {}))
    return matrices, audit


def l2_normalize_rows_inplace(matrix: np.ndarray) -> int:
    # Algebraically the same L2 normalisation. Rescaling before squaring avoids
    # overflow/underflow for Q28's large frequency counts; no frequency is dropped.
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite raw bag vector")
    zero_count = 0
    for start in range(0, len(matrix), 256):
        block = matrix[start:start + 256]
        maxima = np.max(np.abs(block), axis=1)
        nonzero = maxima > 0
        zero_count += int((~nonzero).sum())
        maxima[~nonzero] = 1.0
        block /= maxima[:, None]
        norms = np.linalg.norm(block, axis=1)
        norms[~nonzero] = 1.0
        block /= norms[:, None]
    return zero_count


def score_cosine_1nn(fit: np.ndarray, evaluated: np.ndarray, batch_rows: int, k_neighbors: int = 1) -> np.ndarray:
    if len(fit) == 0 or batch_rows < 1 or k_neighbors not in (1, 5):
        raise ValueError("Nonempty FIT and positive batch size are required")
    scores = np.empty(len(evaluated), dtype=np.float32)
    for start in range(0, len(evaluated), batch_rows):
        similarities = evaluated[start:start + batch_rows] @ fit.T
        if k_neighbors == 1:
            scores[start:start + batch_rows] = np.clip(1.0 - similarities.max(axis=1), 0.0, 2.0)
        else:
            k = min(k_neighbors, len(fit))
            top = np.partition(similarities, len(fit) - k, axis=1)[:, -k:]
            scores[start:start + batch_rows] = np.clip(1.0 - np.mean(top, axis=1, dtype=np.float64), 0.0, 2.0)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite scores")
    return scores


def compute_threshold_bootstrap(calib_scores: np.ndarray, percentile: int,
                                B: int, seed: int) -> tuple:
    if B == 0:
        return float(np.percentile(calib_scores, percentile)), None, None
    rng = np.random.default_rng(seed)
    thresholds = np.zeros(B, dtype=np.float64)
    for i in range(B):
        thresholds[i] = np.percentile(rng.choice(calib_scores, size=len(calib_scores), replace=True), percentile)
    return (float(np.mean(thresholds)), float(np.percentile(thresholds, 2.5)),
            float(np.percentile(thresholds, 97.5)))


def evaluate_scores(calibration: np.ndarray, normal: np.ndarray, anomaly: np.ndarray, config: dict) -> dict:
    scores = np.concatenate([normal, anomaly])
    y = np.concatenate([np.zeros(len(normal), dtype=np.int8), np.ones(len(anomaly), dtype=np.int8)])
    result = {"ap": float(average_precision_score(y, scores)), "auc": float(roc_auc_score(y, scores))}
    for label, values in (("calib", calibration), ("test", scores)):
        for stat, func in (("min", np.min), ("median", np.median), ("max", np.max)):
            result[f"{label}_score_{stat}"] = float(func(values))
    for pct in config["percentiles"]:
        threshold, low, high = compute_threshold_bootstrap(calibration, pct, config["bootstrap_b"], config["seed"])
        prediction = scores > threshold
        tp = int(np.count_nonzero(prediction & (y == 1)))
        fp = int(np.count_nonzero(prediction & (y == 0)))
        fn, tn = len(anomaly) - tp, len(normal) - fp
        suffix = f"_pct{pct}"
        values = {"threshold": threshold, "thr_ci_lo": low, "thr_ci_hi": high,
                  "precision": tp / (tp + fp) if tp + fp else 0.0,
                  "recall": tp / len(anomaly), "f1": 2 * tp / (2 * tp + fp + fn),
                  "n_pred": tp + fp, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                  "fpr": fp / len(normal)}
        result.update({key + suffix: value for key, value in values.items()})
    return result


def atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, data) -> None:
    raw = json.dumps(data, sort_keys=True, indent=2, allow_nan=False).encode()
    atomic_write(path, lambda handle: handle.write(raw))


def output_paths(job: dict) -> dict:
    root = Path(job["output_root"]) / job["anomaly_class"]
    name = f"q{job['query']}_{job['anomaly_class'].lower()}_{job['scale']['name']}_{job['feature']}_countsketch{job['dim']}"
    return {"result": root / "WorkerResults" / (name + ".json"),
            "error": root / "Errors" / (name + ".json"),
            "log": root / "Logs" / (name + ".log"),
            "scores": root / "Scores" / (name + "_scores.npz")}


def peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(raw / (1024**2 if sys.platform == "darwin" else 1024))


def run_job(job: dict) -> dict:
    started = time.perf_counter()
    load_average_start = os.getloadavg()[0]
    assert_inputs_unchanged(job)
    manifest_start = time.perf_counter()
    manifest = load_manifest(Path(job["manifest"]["path"]), job["query"], job["scale"], job["config"]["calib_frac"])
    if manifest != job["manifest"]:
        raise ValueError("Manifest changed between preflight and evaluation")
    manifest_s = time.perf_counter() - manifest_start
    load_start = time.perf_counter()
    matrices, audit = load_matrices(job)
    load_s = time.perf_counter() - load_start
    normalization_start = time.perf_counter()
    zeros = {}
    for name in matrices:
        zeros[f"{name}_zero_vectors"] = l2_normalize_rows_inplace(matrices[name])
        matrices[name] = matrices[name].astype(np.float32)
    normalization_s = time.perf_counter() - normalization_start
    scores, timings = {}, {}
    with threadpool_limits(limits=job["config"]["blas_threads"]):
        blas = threadpool_info()
        for name in ("calib", "test_normal", "test_anom"):
            print(f"[SCORE] {name}: {len(matrices[name]):,} x {len(matrices['fit']):,}; dim={job['dim']}", flush=True)
            score_start = time.perf_counter()
            scores[name] = score_cosine_1nn(matrices["fit"], matrices[name], job["config"]["score_batch_rows"], job["config"]["neighbors"])
            timings[name] = time.perf_counter() - score_start
    metrics_start = time.perf_counter()
    metrics = evaluate_scores(scores["calib"], scores["test_normal"], scores["test_anom"], job["config"])
    metrics_s = time.perf_counter() - metrics_start
    assert_inputs_unchanged(job)
    splits = job["manifest"]["splits"]
    normal_n, anom_n = len(matrices["test_normal"]), len(matrices["test_anom"])
    npz = {"calibration": scores["calib"], "test_normal": scores["test_normal"],
           "test_anomaly": scores["test_anom"], "scores": np.concatenate([scores["test_normal"], scores["test_anom"]]),
           "y_true": np.concatenate([np.zeros(normal_n, dtype=np.int8), np.ones(anom_n, dtype=np.int8)]),
           "test_identity": np.asarray(splits["test_normal"]["identities"] + splits["test_anom"]["identities"]),
           "test_subtype": np.asarray(splits["test_normal"]["subtypes"] + splits["test_anom"]["subtypes"]),
           "calibration_identity": np.asarray(splits["calib"]["identities"]),
           "fit_identity": np.asarray(splits["fit"]["identities"]),
           "job_signature": np.asarray(job["signature"])}
    npz["test_parquet_identity"] = np.asarray(splits["test_normal"]["identities"] + [
        parquet_identity(job, job["anomaly_class"].lower(), i) for i in splits["test_anom"]["identities"]])
    paths = output_paths(job)
    save_start = time.perf_counter()
    atomic_write(paths["scores"], lambda handle: np.savez_compressed(handle, **npz))
    scores_hash = sha256_file(paths["scores"])
    save_s = time.perf_counter() - save_start
    score_total = sum(timings.values())
    total_s = load_s + normalization_s + score_total
    row = {**metrics, **audit, **zeros,
           "query": f"q{job['query']}", "query_id": job["query"], "scale": job["scale"]["name"],
           "anomaly_class": job["anomaly_class"], "feature": job["feature"],
           "sketch_dim": job["dim"], "input_column": vector_column(job["feature"], job["dim"]),
           "representation": "countsketch_bag_frequency", "normalization": "l2",
           "normalization_policy": "row_max_abs_rescale_then_l2_float64_then_float32",
           "stored_vector_dtype": "float64", "scoring_vector_dtype": "float32",
           "stored_sketch_budget_bytes_per_execution": job["dim"] * 8,
           "bag_weights": "witness_occurrence_frequency",
           "bag_psd_version": BAG_PSD_VERSION,
           "representation_family": "countsketch_dense_bag", "representation_dim": job["dim"],
           "sketch_budget_bytes_per_execution": job["dim"] * 4,
           "candidate_policy": "exact_all_fit_no_candidates",
           "svd_used": False, "bag_used": True, "method": "knn_distance", "metric": "cosine",
           "knn_k": job["config"]["neighbors"], "k_neighbors": job["config"]["neighbors"], "vector_sketch_seed": 13,
           "train_k": job["scale"]["train_k"], "n_train": job["scale"]["train_k"],
           "n_fit": len(matrices["fit"]), "n_fit_actual": len(matrices["fit"]),
           "n_calib": len(matrices["calib"]), "test_normal_k": normal_n, "n_anom": anom_n,
           "n_test": normal_n + anom_n, "n_test_normal": normal_n, "n_test_anom": anom_n,
           "n_anom_test_config": job["scale"]["n_anom_test"],
           "contamination": anom_n / (normal_n + anom_n),
           "calib_frac": job["config"]["calib_frac"], "bootstrap_b": job["config"]["bootstrap_b"],
           "threshold_percentiles": ",".join(map(str, job["config"]["percentiles"])),
           "threshold_policy": "mean_bootstrap_calibration_percentile_strict_greater",
           "uses_shared_split": True, "split_policy": "strict_existing_kmv_identity_and_order",
           "shared_split_file": job["manifest"]["path"], "split_sha256": job["manifest"]["sha256"],
           "shared_split_dir": str(Path(job["manifest"]["path"]).parent),
           "split_manifest": job["manifest"]["path"],
           "split_members_and_order_sha256": digest_object({s: splits[s]["identities"] for s in SPLITS}),
           "input_root": job["input_root"], "input_fingerprint": digest_object(job["pools"]),
           "input_fingerprint_policy": "main_and_sidecar_names_sizes_mtime_ns_and_both_success_marker_sha256",
           "code_version": CODE_VERSION, "code_sha256": sha256_file(Path(__file__)),
           "computation_fingerprint": job["computation_fingerprint"], "job_signature": job["signature"],
           "load_s": load_s, "normalize_s": normalization_s, "fit_s": 0.0,
           "manifest_s": manifest_s, "representation_build_s": 0.0,
           "preprocess_s": load_s + normalization_s, "preprocess_total_s": load_s + normalization_s,
           "score_calib_s": timings["calib"], "score_normal_s": timings["test_normal"],
           "score_anom_s": timings["test_anom"], "score_total_s": score_total, "score_s": score_total,
           "model_score_s": score_total, "total_s": total_s, "runtime_s": total_s,
           "metrics_s": metrics_s, "save_scores_s": save_s,
           "end_to_end_s": time.perf_counter() - started,
           "mean_test_scoring_ms": 1000 * (timings["test_normal"] + timings["test_anom"]) / (normal_n + anom_n),
           "cost_protocol": "load_stored_sketches_plus_l2_plus_all_fit_cosine_knn; materialization_excluded",
           "runtime_comparison_mode": "stored_bag_countsketch_l2_cosine_knn",
           "load_protocol": "single_projected_bag_sidecar_pass_per_pool_with_strict_identity_selection",
           "quality_protocol": "shared_split_threshold_from_calib_percentile",
           "comparison_ready": True,
           "comparison_requirements": "same_class_query_scale_splits; timing_requires_matched_hardware_and_load",
           "end_to_end_protocol": "worker_input_checks_load_l2_scoring_metrics_score_export; master_preflight_excluded",
           "mean_test_scoring_protocol": "batched_test_scoring_time_per_execution_not_online_latency",
           "peak_rss_mb": peak_rss_mb(), "peak_rss_unit": "MiB", "peak_rss_scope": "fresh_worker_process",
           "matrix_total_kb": sum(x.nbytes for x in matrices.values()) / 1024,
           "representation_total_kb": sum(x.nbytes for x in matrices.values()) / 1024,
           "zero_vector_policy": "cosine_similarity_zero_including_two_zero_vectors",
           "blas_threads": job["config"]["blas_threads"], "blas_info": json.dumps(blas, sort_keys=True),
           "python_version": platform.python_version(), "numpy_version": np.__version__,
           "sklearn_version": sklearn.__version__, "scipy_version": scipy.__version__,
           "pyarrow_version": pa.__version__, "hostname": platform.node(), "platform": platform.platform(),
           "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
           "load_average_1m_start": load_average_start, "load_average_1m_end": os.getloadavg()[0],
           "parquet_batch_size": job["config"]["parquet_batch_rows"],
           "score_batch_rows": job["config"]["score_batch_rows"],
           "output_dir": str(Path(job["output_root"]) / job["anomaly_class"]),
           "scores_file": str(paths["scores"]), "scores_sha256": scores_hash,
           "log_file": str(paths["log"]), "status": "ok", "error": ""}
    if job.get("identity_translations"):
        row["split_policy"] = "existing_kmv_members_and_order_verified_numeric_id_translation"
        row["load_protocol"] = "single_projected_bag_sidecar_pass_per_pool_with_verified_identity_translation"
    for name, matrix in matrices.items():
        row[f"x{name}_shape"] = str(matrix.shape)
        row[f"x{name}_kb"] = matrix.nbytes / 1024
        row[f"x{name}_dtype"] = str(matrix.dtype)
        row[f"{name}_identities_sha256"] = digest_object(splits[name]["identities"])
    row["xtest_shape"] = str((normal_n + anom_n, job["dim"]))
    row["xtest_kb"] = (matrices["test_normal"].nbytes + matrices["test_anom"].nbytes) / 1024
    write_json(paths["result"], {"signature": job["signature"], "result": row})
    return row


def validate_saved_selection(job: dict, payload: dict, path: Path) -> None:
    splits, row = job["manifest"]["splits"], payload["result"]
    if row["split_sha256"] != job["manifest"]["sha256"] or row["job_signature"] != payload["signature"]:
        raise ValueError(f"Saved result split/signature mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        expected = {"fit_identity": splits["fit"]["identities"],
                    "calibration_identity": splits["calib"]["identities"],
                    "test_identity": splits["test_normal"]["identities"] + splits["test_anom"]["identities"],
                    "test_subtype": splits["test_normal"]["subtypes"] + splits["test_anom"]["subtypes"],
                    "y_true": [0] * len(splits["test_normal"]["identities"]) + [1] * len(splits["test_anom"]["identities"])}
        for key, values in expected.items():
            if not np.array_equal(saved[key], np.asarray(values)):
                raise ValueError(f"Saved {key} differs from original split membership/order: {path}")
        if str(saved["job_signature"].item()) != payload["signature"]:
            raise ValueError(f"Saved score signature mismatch: {path}")
        if job.get("identity_translations") or "test_parquet_identity" in saved.files:
            actual = splits["test_normal"]["identities"] + [parquet_identity(job, job["anomaly_class"].lower(), i)
                                                          for i in splits["test_anom"]["identities"]]
            if not np.array_equal(saved["test_parquet_identity"], np.asarray(actual)):
                raise ValueError(f"Saved physical Parquet identity mismatch: {path}")
        for key, split in (("calibration", "calib"), ("test_normal", "test_normal"), ("test_anomaly", "test_anom")):
            if saved[key].shape != (len(splits[split]["identities"]),) or not np.isfinite(saved[key]).all():
                raise ValueError(f"Invalid saved {key} scores: {path}")
        scores = np.concatenate([saved["test_normal"], saved["test_anomaly"]])
        if not np.array_equal(saved["scores"], scores):
            raise ValueError(f"Saved scores have different order/content: {path}")
        for key, function in (("ap", average_precision_score), ("auc", roc_auc_score)):
            if not np.isclose(function(saved["y_true"], scores), row[key], rtol=0, atol=1e-12):
                raise ValueError(f"Saved {key} disagrees with saved scores/labels: {path}")


def read_completed(job: dict) -> dict | None:
    paths = output_paths(job)
    if not paths["result"].exists():
        return None
    payload = json.loads(paths["result"].read_text())
    signature_matches = payload.get("signature") == job["signature"]
    if not signature_matches or payload.get("result", {}).get("status") != "ok":
        raise ValueError(f"Existing result has different inputs/settings: {paths['result']}. "
                         "Nothing was overwritten. Use a separate OUT_DIR for a different experiment.")
    row = payload["result"]
    if not paths["scores"].is_file() or sha256_file(paths["scores"]) != row["scores_sha256"]:
        raise ValueError(f"Saved score file missing or changed: {paths['scores']}")
    validate_saved_selection(job, payload, paths["scores"])
    return row


def child_entry(job: dict) -> None:
    paths = output_paths(job)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    with paths["log"].open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            print(f"[START] {paths['result'].stem}", flush=True)
            row = run_job(job)
            print(f"[OK] AP={row['ap']:.6f}; AUC={row['auc']:.6f}; total_s={row['total_s']:.2f}", flush=True)
        except Exception as exc:
            error = {"status": "error", "query": f"q{job['query']}", "scale": job["scale"]["name"],
                     "feature": job["feature"], "sketch_dim": job["dim"], "anomaly_class": job["anomaly_class"],
                     "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                     "log_file": str(paths["log"]), "job_signature": job["signature"]}
            write_json(paths["error"], error)
            traceback.print_exc()
            raise SystemExit(1)


def build_jobs() -> list[dict]:
    if (not ANOMALY_CLASSES or set(ANOMALY_CLASSES) - {"S1", "S2"} or not QUERY_IDS
            or not FEATURES or set(FEATURES) - set(ALGORITHMS) or not SKETCH_DIMS
            or set(SKETCH_DIMS) - {512, 1024}):
        raise ValueError("Invalid query/class/feature/dimension selection")
    if (not 0 < CALIB_FRAC < 1 or BOOTSTRAP_B < 0 or BLAS_THREADS < 1 or SCORE_BATCH_ROWS < 1
            or PARQUET_BATCH_ROWS < 1 or MAX_MATRIX_GIB <= 0 or WORKER_TIMEOUT_SECONDS <= 0):
        raise ValueError("Invalid resource/evaluation configuration")
    for selection in (ANOMALY_CLASSES, QUERY_IDS, FEATURES, SKETCH_DIMS, [s["name"] for s in SCALES]):
        if len(selection) != len(set(selection)):
            raise ValueError("Duplicate configured query/class/feature/dimension/scale")
    if not SCALES or not THRESHOLD_PERCENTILES or any(not 0 < p < 100 for p in THRESHOLD_PERCENTILES):
        raise ValueError("Invalid scales or threshold percentiles")
    inputs = INPUT_ROOT.resolve()
    outputs = OUT_DIR.resolve()
    if inputs == outputs or inputs in outputs.parents or outputs in inputs.parents:
        raise ValueError("OUT_DIR must be separate from INPUT_ROOT")
    for anomaly_class in ANOMALY_CLASSES:
        directory = split_directory(anomaly_class).resolve()
        if outputs == directory or outputs in directory.parents or directory in outputs.parents:
            raise ValueError("OUT_DIR must be separate from the split directories")
        if not directory.is_dir():
            raise FileNotFoundError(f"Existing KMV split folder missing: {directory}. It will not be created.")
    config = {"calib_frac": CALIB_FRAC, "bootstrap_b": BOOTSTRAP_B, "seed": RANDOM_SEED,
              "percentiles": list(THRESHOLD_PERCENTILES), "blas_threads": BLAS_THREADS,
              "score_batch_rows": SCORE_BATCH_ROWS, "parquet_batch_rows": PARQUET_BATCH_ROWS,
              "max_matrix_gib": MAX_MATRIX_GIB, "neighbors": K_NEIGHBORS}
    code = computation_fingerprint()
    manifests = {}
    for cls in ANOMALY_CLASSES:
        for query in QUERY_IDS:
            for scale in SCALES:
                path = split_path(cls, query, scale["name"])
                manifests[(cls, query, scale["name"])] = load_manifest(path, query, scale, CALIB_FRAC)
    jobs = []
    for query in QUERY_IDS:
        normal = inspect_pool(query, "normal")
        resolve_pool_identities(query, "normal", normal, [v for (c, q, s), v in manifests.items() if q == query])
        for cls in ANOMALY_CLASSES:
            pools = {"normal": normal, cls.lower(): inspect_pool(query, cls.lower())}
            translations = resolve_pool_identities(query, cls.lower(), pools[cls.lower()],
                                                   [v for (c, q, s), v in manifests.items() if q == query and c == cls])
            for scale in SCALES:
                manifest = manifests[(cls, query, scale["name"])]
                selected_translations = {i: translations[i] for i in manifest["splits"]["test_anom"]["identities"]
                                         if i in translations}
                for feature in FEATURES:
                    for dim in SKETCH_DIMS:
                        job = {"query": query, "anomaly_class": cls, "scale": scale, "feature": feature,
                               "dim": dim, "config": config, "pools": pools, "manifest": manifest,
                               "input_root": str(inputs), "output_root": str(outputs), "computation_fingerprint": code}
                        if selected_translations:
                            job["identity_translations"] = {cls.lower(): selected_translations}
                        job["signature"] = digest_object({k: v for k, v in job.items() if k != "output_root"})
                        read_completed(job)
                        jobs.append(job)
    return jobs


def aggregate_outputs(root: Path) -> None:
    results, errors = [], []
    for cls in ("S1", "S2"):
        folder = root / cls
        class_results = [json.loads(p.read_text())["result"] for p in sorted((folder / "WorkerResults").glob("*.json"))]
        class_errors = [json.loads(p.read_text()) for p in sorted((folder / "Errors").glob("*.json"))
                        if not (folder / "WorkerResults" / p.name).exists()]
        results.extend(class_results)
        errors.extend(class_errors)
        if folder.is_dir():
            write_csv(folder / f"bag_countsketch_{cls.lower()}_results_all.csv", class_results)
            write_csv(folder / f"bag_countsketch_{cls.lower()}_errors.csv", class_errors)
    write_csv(root / "bag_countsketch_s1_s2_results_all.csv", results)
    write_csv(root / "bag_countsketch_s1_s2_errors.csv", errors)


def write_csv(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    fields = sorted({key for row in rows for key in row}) or ["status", "error"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    raw = buffer.getvalue().encode()
    atomic_write(path, lambda handle: handle.write(raw))


def save_input_audit(jobs: list[dict]) -> None:
    # Archive inputs once per pool/split, not once for every vector dimension.
    root = Path(jobs[0]["output_root"]) / "InputAudit"
    seen_pools, seen_splits = set(), set()
    resume_audit = []
    for job in jobs:
        for pool, snapshot in job["pools"].items():
            key = (job["query"], pool)
            if key not in seen_pools:
                write_json(root / "Pools" / f"q{key[0]}_{key[1]}.json", snapshot)
                seen_pools.add(key)
        source = Path(job["manifest"]["path"])
        if str(source) not in seen_splits:
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != job["manifest"]["sha256"]:
                raise ValueError(f"Split changed before audit: {source}")
            atomic_write(root / job["anomaly_class"] / source.name, lambda handle: handle.write(raw))
            translations = job.get("identity_translations", {}).get(job["anomaly_class"].lower(), {})
            if translations:
                write_csv(root / "IdentityTranslations" / job["anomaly_class"] / source.name,
                          [{"split_identity": identity, **proof} for identity, proof in sorted(translations.items())])
            seen_splits.add(str(source))
        row = read_completed(job)
        resume_audit.append({"query": f"q{job['query']}", "anomaly_class": job["anomaly_class"],
                             "scale": job["scale"]["name"], "feature": job["feature"], "dim": job["dim"],
                             "status": "VERIFIED_SAVED_SPLITS_LABELS_AP_AUC" if row else "PENDING",
                             "split_sha256": job["manifest"]["sha256"],
                             "result_file": str(output_paths(job)["result"]),
                             "scores_sha256": row["scores_sha256"] if row else ""})
    write_csv(root / "resume_validation.csv", resume_audit)
    raw_code = Path(__file__).read_bytes()
    atomic_write(root / ("evaluator_" + hashlib.sha256(raw_code).hexdigest() + ".py"),
                 lambda handle: handle.write(raw_code))


def launch_worker(job: dict) -> int:
    process = multiprocessing.get_context("spawn").Process(target=child_entry, args=(job,))
    process.start()
    started = time.monotonic()
    try:
        while process.is_alive():
            process.join(timeout=30)
            elapsed = time.monotonic() - started
            if process.is_alive():
                print(f"  [RUNNING] {elapsed:.0f}s; log={output_paths(job)['log']}", flush=True)
            if process.is_alive() and elapsed > WORKER_TIMEOUT_SECONDS:
                raise TimeoutError(f"Configuration exceeded {WORKER_TIMEOUT_SECONDS}s")
        return int(process.exitcode)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()


def main() -> int:
    print("BAG COUNTSKETCH: stored frequencies -> L2 -> exhaustive cosine k-NN; no PSD, no SVD.", flush=True)
    print(f"INPUT_ROOT={INPUT_ROOT}\nSPLIT_DIRS={SPLIT_DIRS}\nOUT_DIR={OUT_DIR}", flush=True)
    print("Preflight checks ALL split identities against Parquets and verifies saved results before scoring.", flush=True)
    jobs = build_jobs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    with (OUT_DIR / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_input_audit(jobs)
        try:
            for index, job in enumerate(jobs, 1):
                paths = output_paths(job)
                assert_inputs_unchanged(job)
                if read_completed(job) is not None:
                    print(f"[RESUME {index}/{len(jobs)}] {paths['result'].stem}", flush=True)
                    continue
                print(f"[RUN {index}/{len(jobs)}] {paths['result'].stem}", flush=True)
                try:
                    exitcode = launch_worker(job)
                    row = read_completed(job)
                    if exitcode != 0 or row is None:
                        raise RuntimeError(f"Worker exit={exitcode}; see {paths['log']}")
                    print(f"[OK] AP={row['ap']:.6f}; AUC={row['auc']:.6f}; score={row['score_total_s']:.2f}s", flush=True)
                except (RuntimeError, TimeoutError) as exc:
                    failures += 1
                    if not paths["error"].exists():
                        write_json(paths["error"], {"status": "error", "query": f"q{job['query']}",
                                   "anomaly_class": job["anomaly_class"], "scale": job["scale"]["name"],
                                   "feature": job["feature"], "sketch_dim": job["dim"],
                                   "error_type": type(exc).__name__, "error": str(exc), "log_file": str(paths["log"])})
                    print(f"[FAILED] {exc}", flush=True)
                aggregate_outputs(OUT_DIR)
        finally:
            aggregate_outputs(OUT_DIR)
    print(f"[DONE] configurations={len(jobs)}; failures={failures}; results={OUT_DIR}", flush=True)
    return int(failures > 0)



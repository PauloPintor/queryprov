# PSD evaluation extracted from Eval_Scalars_NoCount.py.
from __future__ import annotations


import csv


import hashlib


import json


import os


import platform


import resource


import subprocess


import sys


import time


import traceback


from collections import Counter


from contextlib import redirect_stderr, redirect_stdout


from pathlib import Path


from typing import Any, Dict, List, Sequence, Tuple


import numpy as np


import pyarrow


import pyarrow.parquet as pq


import sklearn


from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)


from sklearn.neighbors import NearestNeighbors


from sklearn.preprocessing import StandardScaler


from threadpoolctl import threadpool_info


QUERY_IDS = (3, 5, 18, 20, 28, 60, 66, 82, 84, 85, 89, 92, 93, 94, 98)


ANOMALY_CLASSES = ("S1", "S2")


SCALES = (
    {"name": "5k", "train_k": 5000, "test_normal_k": 1250, "n_anom_test": 66},
    {"name": "10k", "train_k": 10000, "test_normal_k": 2500, "n_anom_test": 125},
    #{"name": "25k", "train_k": 25000, "test_normal_k": 6250, "n_anom_test": 313},
    #{"name": "50k", "train_k": 50000, "test_normal_k": 12500, "n_anom_test": 625},
)


UNSUP_METHODS = ("knn_distance", )


DISTANCE_METRICS = ("cosine",)


RANDOM_SEED = 13


CALIB_FRAC = 0.20


KNN_K = 5


LOF_N_NEIGHBORS = 1


THRESHOLD_PERCENTILES = (95, 99)


BOOTSTRAP_B = 1000


PARQUET_BATCH_SIZE = 4096


WORKER_TIMEOUT_S = 5 * 60 * 60


RERUN_COMPLETED = False


RERUN_TIMEOUTS = False


REQUIRE_COMPLETE_SIDECARS = True


ALLOW_UNIQUE_NAME_ID_TRANSLATION = True


SIDECAR_SUFFIX = ".bagcompact.parquet"


SCALAR_COLUMNS = ("nrows", "nwitness_unique", "nwitness_total", "ntokens_unique", "ntokens_total")


STORED_SCALAR_FEATURE_NAMES = SCALAR_COLUMNS + (
    "token_unique_ratio", "witness_unique_ratio", "witness_token_ratio",
)


SCALAR_FEATURE_NAMES = STORED_SCALAR_FEATURE_NAMES[1:]


SCALAR_DIM = len(SCALAR_FEATURE_NAMES)


IDENTITY_COLUMNS = ("query_id", "query_name", "subtype")


SOURCE_METADATA = ("psd_population", "psd_counts_exact", "psd_method", "bag_psd_version")


CODE_VERSION = "2026-11-10-psd-optional-cardinality-knn-cli"


SPLIT_NAMES = ("train_normal", "fit", "calib", "test_normal", "test_anom")


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0 * 1024.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def ratio(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) > 0.0 else 0.0


def scalar_features_from_row(row: Dict[str, Any]) -> List[float]:
    """
    Seven provenance descriptors; optional result cardinality is prepended:

    1. nwitness_unique
    2. nwitness_total
    3. ntokens_unique
    4. ntokens_total
    5. ntokens_unique / ntokens_total
    6. nwitness_unique / nwitness_total
    7. nwitness_unique / ntokens_unique
    """

    nwitness_unique = _safe_float(row.get("nwitness_unique"))
    nwitness_total = _safe_float(row.get("nwitness_total"))
    ntokens_unique = _safe_float(row.get("ntokens_unique"))
    ntokens_total = _safe_float(row.get("ntokens_total"))

    return ([_safe_float(row.get("nrows"))] if INCLUDE_NROWS else []) + [
        float(nwitness_unique),
        float(nwitness_total),
        float(ntokens_unique),
        float(ntokens_total),
        ratio(ntokens_unique, max(1.0, ntokens_total)),
        ratio(nwitness_unique, max(1.0, nwitness_total)),
        ratio(nwitness_unique, max(1.0, ntokens_unique)),
    ]


def fit_knn(Xfit: Any, metric: str) -> NearestNeighbors:
    k = min(
        KNN_K,
        max(1, Xfit.shape[0]),
    )

    model = NearestNeighbors(
        n_neighbors=k,
        metric=metric,
        algorithm="brute",
        n_jobs=N_JOBS,
    )

    model.fit(Xfit)

    return model


def score_knn(
    model: NearestNeighbors,
    X: Any,
) -> np.ndarray:
    dists, _ = model.kneighbors(
        X,
        return_distance=True,
    )

    return np.asarray(dists[:, 0] if KNN_K == 1 else np.mean(dists, axis=1, dtype=np.float64), dtype=np.float32)


def safe_ap(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan")


def safe_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")

        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def compute_threshold_bootstrap(
    calib_scores: np.ndarray,
    percentile: int,
    B: int,
    seed: int,
) -> Tuple[float, float, float]:
    if len(calib_scores) == 0:
        return float("nan"), float("nan"), float("nan")

    if B <= 0:
        thr = float(np.percentile(calib_scores, percentile))
        return thr, float("nan"), float("nan")

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


def evaluate_scores(
    y_true: np.ndarray,
    calib_scores: np.ndarray,
    test_scores: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ap": safe_ap(y_true, test_scores),
        "auc": safe_auc(y_true, test_scores),

        "calib_score_min": float(np.min(calib_scores)),
        "calib_score_median": float(np.median(calib_scores)),
        "calib_score_max": float(np.max(calib_scores)),

        "test_score_min": float(np.min(test_scores)),
        "test_score_median": float(np.median(test_scores)),
        "test_score_max": float(np.max(test_scores)),
    }

    for pct in THRESHOLD_PERCENTILES:
        thr, ci_lo, ci_hi = compute_threshold_bootstrap(
            calib_scores=calib_scores,
            percentile=pct,
            B=BOOTSTRAP_B,
            seed=RANDOM_SEED,
        )

        pred = (test_scores > thr).astype(np.int32)

        suffix = f"_pct{pct}"

        out[f"threshold{suffix}"] = float(thr)
        out[f"thr_ci_lo{suffix}"] = float(ci_lo)
        out[f"thr_ci_hi{suffix}"] = float(ci_hi)

        out[f"f1{suffix}"] = float(f1_score(y_true, pred, zero_division=0))
        out[f"precision{suffix}"] = float(precision_score(y_true, pred, zero_division=0))
        out[f"recall{suffix}"] = float(recall_score(y_true, pred, zero_division=0))
        out[f"n_pred{suffix}"] = int(pred.sum())

    return out


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_hash(items):
    return digest_object(items)


def array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    if not fields:
        fields = ["anomaly_class", "query", "scale", "feature", "method", "metric", "status", "error"]
    temp = path.with_name("." + path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def environment():
    return {
        "hostname": platform.node(), "platform": platform.platform(),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__, "pyarrow_version": pyarrow.__version__,
        "logical_cpu_count": os.cpu_count(), "blas_info": threadpool_info(),
    }


def split_path(root, anomaly_class, query, scale):
    suffix = "_s2" if anomaly_class == "S2" else ""
    return Path(root) / f"q{query}_{scale}{suffix}_shared_split_ids.csv"


def read_manifest(path, query, scale):
    positioned = {name: [] for name in SPLIT_NAMES}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {"query", "scale", "split", "pos", "identity", "subtype"}
        if not needed.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing split columns {sorted(needed-set(reader.fieldnames or ())) }")
        for row in reader:
            if row["query"].lower() != f"q{query}" or row["scale"] != scale["name"]:
                raise ValueError(f"{path}: unexpected query or scale")
            name = row["split"]
            if name not in positioned:
                raise ValueError(f"{path}: unknown split {name}")
            identity = row["identity"]
            execution_id, query_name = identity.split("|", 1)
            if int(execution_id) < 0 or not query_name or query_name.startswith("._"):
                raise ValueError(f"{path}: invalid identity {identity}")
            positioned[name].append((int(row["pos"]), row))
    manifest = {}
    for name, records in positioned.items():
        records.sort(key=lambda item: item[0])
        if [p for p, _ in records] != list(range(len(records))):
            raise ValueError(f"{path}: non-contiguous positions in {name}")
        manifest[name] = [row for _, row in records]
        ids = [row["identity"] for row in manifest[name]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate identities in {name}")
    sets = {name: {r["identity"] for r in records} for name, records in manifest.items()}
    if sets["train_normal"] != sets["fit"] | sets["calib"] or sets["fit"] & sets["calib"]:
        raise ValueError(f"{path}: train_normal must be disjoint fit + calib")
    if sets["train_normal"] & sets["test_normal"]:
        raise ValueError(f"{path}: normal training/test overlap")
    expected = {
        "train_normal": scale["train_k"], "fit": int(scale["train_k"] * (1-CALIB_FRAC)),
        "calib": int(scale["train_k"] * CALIB_FRAC),
        "test_normal": scale["test_normal_k"], "test_anom": scale["n_anom_test"],
    }
    for name, size in expected.items():
        if len(manifest[name]) != size:
            raise ValueError(f"{path}: {name}={len(manifest[name])}, expected {size}")
    return manifest


def source_files(directory):
    directory = Path(directory)
    paths = sorted(path for path in directory.rglob("*" + SIDECAR_SUFFIX)
                   if not any(part.startswith(".") for part in path.relative_to(directory).parts))
    if not paths:
        raise FileNotFoundError(f"No scalar sidecars: {directory}")
    if REQUIRE_COMPLETE_SIDECARS and not (directory / "_BAG_PSD_SUCCESS.json").is_file():
        raise FileNotFoundError(f"Missing completion marker: {directory / '_BAG_PSD_SUCCESS.json'}")
    main = sorted(path for path in directory.rglob("*.parquet")
                  if not any(part.startswith(".") for part in path.relative_to(directory).parts)
                  and not path.name.endswith((SIDECAR_SUFFIX, ".bagtfcompact.parquet")))
    expected = {str(p.relative_to(directory))[:-len(".parquet")] + SIDECAR_SUFFIX for p in main}
    actual = {str(p.relative_to(directory)) for p in paths}
    if expected != actual:
        raise ValueError(f"Main/sidecar batch mismatch in {directory}: "
                         f"missing={sorted(expected-actual)[:5]}, extra={sorted(actual-expected)[:5]}")
    return paths


def pool_fingerprint(directory, files):
    directory = Path(directory)
    records = []
    for path in files:
        stat = path.stat()
        records.append((str(path.relative_to(directory)), stat.st_size, stat.st_mtime_ns))
    marker = directory / "_BAG_PSD_SUCCESS.json"
    return digest_object({"files": records, "completion_marker": sha256_file(marker) if marker.exists() else None})


def validate_scalar_row(row):
    for name in SCALAR_COLUMNS:
        value = row.get(name)
        if value is None or not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Invalid stored scalar {name}: {value!r}")
    if float(row["nwitness_unique"]) > float(row["nwitness_total"]):
        raise ValueError("nwitness_unique exceeds nwitness_total")
    if float(row["ntokens_unique"]) > float(row["ntokens_total"]):
        raise ValueError("ntokens_unique exceeds ntokens_total")
    if row.get("psd_feature_names") is not None and tuple(row["psd_feature_names"]) != STORED_SCALAR_FEATURE_NAMES:
        raise ValueError("Stored scalar feature names/order differ")
    vector = np.asarray(scalar_features_from_row(row), dtype=np.float32)
    if vector.shape != (SCALAR_DIM,):
        raise ValueError(f"Scalar feature shape {vector.shape}; expected ({SCALAR_DIM},)")
    if not np.isfinite(vector).all():
        raise ValueError(f"The {SCALAR_DIM}-feature float32 mapping produced non-finite values")
    return vector


def load_pool(directory, requested, expected_fingerprint):
    files = source_files(directory)
    if pool_fingerprint(directory, files) != expected_fingerprint:
        raise ValueError(f"Input changed since preflight: {directory}")
    wanted = {}
    for row in requested:
        name = row["identity"].split("|", 1)[1]
        if name in wanted:
            raise ValueError(f"Query name not unique in requested pool: {name}")
        wanted[name] = row
    found, translations, provenance = {}, [], Counter()
    loaded_files = scanned = 0
    for path in files:
        with pq.ParquetFile(path) as parquet:
            names = set(parquet.schema_arrow.names)
            required = set(IDENTITY_COLUMNS + SCALAR_COLUMNS)
            missing = required - names
            if missing:
                raise ValueError(f"{path}: missing scalar/identity columns {sorted(missing)}")
            columns = list(IDENTITY_COLUMNS + SCALAR_COLUMNS)
            columns += [c for c in (*SOURCE_METADATA, "psd_feature_names") if c in names]
            for batch in parquet.iter_batches(columns=columns, batch_size=PARQUET_BATCH_SIZE, use_threads=True):
                for row in batch.to_pylist():
                    scanned += 1
                    name = str(row["query_name"])
                    if name not in wanted:
                        continue
                    if name in found:
                        raise ValueError(f"Duplicate query name in input pool: {name}")
                    canonical = wanted[name]
                    physical = f"{int(row['query_id'])}|{name}"
                    if physical != canonical["identity"]:
                        if not ALLOW_UNIQUE_NAME_ID_TRANSLATION:
                            raise ValueError(f"Identity differs: {canonical['identity']} -> {physical}")
                        translations.append({"manifest_identity": canonical["identity"],
                                             "parquet_identity": physical, "query_name": name})
                    subtype = str(row["subtype"])
                    if canonical["subtype"] and subtype != canonical["subtype"]:
                        raise ValueError(f"{path}: subtype differs for {name}: {subtype} != {canonical['subtype']}")
                    vector = validate_scalar_row(row)
                    meta = {key: row.get(key) for key in SOURCE_METADATA}
                    provenance[json.dumps(meta, sort_keys=True)] += 1
                    found[name] = {"raw": vector, "identity": canonical["identity"],
                                   "parquet_identity": physical, "subtype": subtype}
        loaded_files += 1
    missing = set(wanted) - set(found)
    if missing:
        raise ValueError(f"{directory}: {len(missing)} split executions missing; examples={sorted(missing)[:5]}")
    if pool_fingerprint(directory, source_files(directory)) != expected_fingerprint:
        raise ValueError(f"Input changed while reading: {directory}")
    report = {"files_found": len(files), "files_read": loaded_files, "rows_seen": scanned,
              "rows_loaded": len(found), "files_missing_feature": 0, "files_with_error": 0,
              "rows_invalid": 0, "tokens_loaded": 0}
    return found, translations, provenance, report


def ordered_values(pool, entries):
    values = [pool[r["identity"].split("|", 1)[1]] for r in entries]
    return np.stack([r["raw"] for r in values]).astype(np.float32), values


def matrix_info(matrix, prefix):
    return {f"{prefix}_shape": str(matrix.shape), f"{prefix}_dtype": str(matrix.dtype),
            f"{prefix}_kb": matrix.nbytes / 1024, f"{prefix}_nnz": ""}


def run_job(request):
    started = time.perf_counter()
    cls, query, scale = request["anomaly_class"], request["query_id"], request["scale"]
    manifest_file = Path(request["shared_split_file"])
    if sha256_file(manifest_file) != request["split_sha256"]:
        raise ValueError("Split changed since preflight")
    manifest = read_manifest(manifest_file, query, scale)
    load_start = time.perf_counter()
    normal_requested = [r for name in ("fit", "calib", "test_normal") for r in manifest[name]]
    normals, normal_translations, normal_provenance, normal_report = load_pool(
        Path(request["normal_directory"]), normal_requested, request["normal_input_fingerprint"])
    anomalies, anomaly_translations, anomaly_provenance, anomaly_report = load_pool(
        Path(request["anomaly_directory"]), manifest["test_anom"], request["anomaly_input_fingerprint"])
    load_s = time.perf_counter() - load_start

    split_start = time.perf_counter()
    raw, metadata = {}, {}
    for name in ("fit", "calib", "test_normal", "test_anom"):
        raw[name], metadata[name] = ordered_values(anomalies if name == "test_anom" else normals, manifest[name])
    split_s = time.perf_counter() - split_start

    build_start = time.perf_counter()
    scaler = StandardScaler()
    matrices = {"fit": scaler.fit_transform(raw["fit"]).astype(np.float32)}
    for name in ("calib", "test_normal", "test_anom"):
        matrices[name] = scaler.transform(raw[name]).astype(np.float32)
    if not all(np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("StandardScaler produced non-finite values")
    build_s = time.perf_counter() - build_start

    fit_start = time.perf_counter()
    method, metric = request["method"], request["metric"]
    model = fit_knn(matrices["fit"], metric)
    fit_s = time.perf_counter() - fit_start
    scores, timings = {}, {}
    for name in ("calib", "test_normal", "test_anom"):
        tick = time.perf_counter()
        scores[name] = score_knn(model, matrices[name])
        timings[name] = time.perf_counter() - tick
        if not np.isfinite(scores[name]).all():
            raise ValueError(f"Non-finite {name} scores")
    test_scores = np.concatenate((scores["test_normal"], scores["test_anom"])).astype(np.float32, copy=False)
    y_true = np.concatenate((np.zeros(len(scores["test_normal"]), dtype=np.int8),
                             np.ones(len(scores["test_anom"]), dtype=np.int8)))
    tick = time.perf_counter()
    result = evaluate_scores(y_true, scores["calib"], test_scores)
    metrics_s = time.perf_counter() - tick
    for pct in THRESHOLD_PERCENTILES:
        pred = test_scores > result[f"threshold_pct{pct}"]
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        result.update({f"tp_pct{pct}": tp, f"fp_pct{pct}": fp,
                       f"fn_pct{pct}": int(y_true.sum())-tp,
                       f"tn_pct{pct}": int(np.sum(y_true == 0))-fp,
                       f"fpr_pct{pct}": fp / len(scores["test_normal"])})
    translations = normal_translations + anomaly_translations
    provenance = normal_provenance + anomaly_provenance
    audit_file = Path(request["identity_audit_file"])
    identity_audit = {"job_signature": request["job_signature"], "shared_split_file": str(manifest_file),
                      "split_sha256": request["split_sha256"], "identity_translation_count": len(translations),
                      "translations": translations,
                      "scalar_source_metadata": [{"metadata": json.loads(k), "rows": v} for k, v in provenance.items()],
                      "splits": {name: {"count": len(manifest[name]),
                                        "ordered_identities_sha256": ordered_hash([r["identity"] for r in manifest[name]])}
                                 for name in manifest}}
    write_json(audit_file, identity_audit)

    npz = {"calibration": scores["calib"], "test_normal": scores["test_normal"], "test_anomaly": scores["test_anom"],
           "scores": test_scores, "y_true": y_true,
           "test_identity": np.asarray([r["identity"] for n in ("test_normal", "test_anom") for r in metadata[n]]),
           "test_parquet_identity": np.asarray([r["parquet_identity"] for n in ("test_normal", "test_anom") for r in metadata[n]]),
           "test_subtype": np.asarray([r["subtype"] for n in ("test_normal", "test_anom") for r in metadata[n]]),
           "calibration_identity": np.asarray([r["identity"] for r in metadata["calib"]]),
           "fit_identity": np.asarray([r["identity"] for r in metadata["fit"]]),
           "scalar_feature_names": np.asarray(SCALAR_FEATURE_NAMES), "scaler_mean": scaler.mean_,
           "scaler_scale": scaler.scale_, "scaler_var": scaler.var_,
           "job_signature": np.asarray(request["job_signature"])}
    for name in raw:
        npz[f"{name}_raw_features"] = raw[name]
    save_tick = time.perf_counter()
    score_file = Path(request["scores_file"])
    score_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = score_file.with_name("." + score_file.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **npz)
    os.replace(temporary, score_file)
    save_scores_s = time.perf_counter() - save_tick

    score_total_s = sum(timings.values())
    runtime_s = load_s + split_s + build_s + fit_s + score_total_s
    result.update({
        "anomaly_class": cls, "query": f"q{query}", "query_id": query, "scale": scale["name"],
        "feature": "scalars", "method": method, "metric": metric,
        "representation": f"full_scalar_summary_{SCALAR_DIM}_features_standardized", "representation_family": "full_vector",
        "representation_dim": SCALAR_DIM, "feature_dim": SCALAR_DIM,
        "scalar_feature_names": json.dumps(SCALAR_FEATURE_NAMES),
        "scalar_source": "stored_bagcompact_psd_counts", "psd_variant": "psd-r" if INCLUDE_NROWS else "psd",
        "scalar_source_metadata": json.dumps(identity_audit["scalar_source_metadata"], sort_keys=True),
        "uses_kmv": False, "uses_hashing_vectorizer": False, "uses_svd": False, "uses_tfidf": False,
        "uses_reduction": False, "svd_used": False, "bag_used": False, "uses_shared_split": True,
        "train_k": scale["train_k"], "test_normal_k": scale["test_normal_k"],
        "n_anom_test_config": scale["n_anom_test"], "n_train": len(manifest["train_normal"]),
        "n_fit": len(raw["fit"]), "n_fit_actual": len(raw["fit"]), "n_calib": len(raw["calib"]),
        "n_test_normal": len(raw["test_normal"]), "n_test_anom": len(raw["test_anom"]),
        "n_anom": len(raw["test_anom"]), "n_test": len(y_true), "contamination": float(y_true.mean()),
        "calib_frac": CALIB_FRAC, "knn_k": KNN_K, "k_neighbors": KNN_K,
        "bootstrap_b": BOOTSTRAP_B, "random_seed": RANDOM_SEED, "threshold_percentiles": "95,99",
        "threshold_policy": "mean_bootstrap_calibration_percentile_strict_greater",
        "load_s": load_s, "split_s": split_s, "build_s": build_s, "representation_build_s": build_s,
        "fit_s": fit_s, "score_calib_s": timings["calib"], "score_normal_s": timings["test_normal"],
        "score_anom_s": timings["test_anom"], "score_s": score_total_s, "score_total_s": score_total_s,
        "model_score_s": fit_s + score_total_s, "preprocess_s": load_s + split_s + build_s,
        "preprocess_total_s": load_s + split_s + build_s, "total_s": runtime_s, "runtime_s": runtime_s,
        "metrics_s": metrics_s, "save_scores_s": save_scores_s, "end_to_end_s": time.perf_counter()-started,
        "mean_test_scoring_ms": 1000*(timings["test_normal"]+timings["test_anom"])/len(y_true),
        "mean_test_scoring_protocol": "batched_test_scoring_time_per_execution_not_online_latency",
        "peak_rss_mb": peak_rss_mb(), "peak_rss_unit": "MiB", "peak_rss_scope": "fresh_worker_process",
        "representation_total_kb": sum(x.nbytes for x in matrices.values()) / 1024,
        "matrix_total_kb": sum(x.nbytes for x in matrices.values()) / 1024,
        "input_root": request["input_root"], "shared_split_dir": str(manifest_file.parent),
        "shared_split_file": str(manifest_file), "split_manifest": str(manifest_file),
        "split_sha256": request["split_sha256"],
        "split_members_and_order_sha256": digest_object({n: [r["identity"] for r in manifest[n]] for n in manifest}),
        "identity_translation_count": len(translations), "identity_translation_sha256": digest_object(translations),
        "identity_audit_file": str(audit_file),
        "split_policy": "existing_members_and_order_exact_identity_or_unique_pool_query_name",
        "input_fingerprint": request["input_fingerprint"],
        "input_fingerprint_policy": "sidecar_names_sizes_mtime_ns_and_bag_psd_success_marker_sha256",
        "load_protocol": f"project_stored_scalar_counts_then_{SCALAR_DIM}_features_manifest_order",
        "quality_protocol": "shared_split_threshold_from_calib_percentile",
        "cost_protocol": f"sidecar_load_plus_{SCALAR_DIM}_scalar_mapping_plus_standardscaler_plus_detector; materialization_excluded",
        "runtime_comparison_mode": "scalar_worker_isolated",
        "code_version": CODE_VERSION, "code_sha256": request["code_sha256"], "job_signature": request["job_signature"],
        "scores_file": str(score_file), "scores_sha256": sha256_file(score_file),
        "log_file": request["log_file"], "comparison_ready": True, "status": "ok", "error": "",
        **environment(),
    })
    for name, prefix in (("fit", "xfit"), ("calib", "xcalib"), ("test_normal", "xtest_normal"),
                         ("test_anom", "xtest_anom")):
        result.update(matrix_info(matrices[name], prefix))
        result[f"{name}_identities_sha256"] = ordered_hash([r["identity"] for r in metadata[name]])
        result[f"{name}_raw_vectors_sha256"] = array_hash(raw[name])
    result["xtest_shape"] = str((len(y_true), SCALAR_DIM))
    result["xtest_kb"] = (matrices["test_normal"].nbytes + matrices["test_anom"].nbytes) / 1024
    for prefix, report in (("normal", normal_report), ("anomaly", anomaly_report)):
        result.update({f"{prefix}_{key}": value for key, value in report.items()})
    if sha256_file(manifest_file) != request["split_sha256"]:
        raise ValueError("Split changed during evaluation")
    return result


def worker(request_file):
    request = json.loads(Path(request_file).read_text())
    from .psd_cli import configure
    configure(request["cli_config"])
    log = Path(request["log_file"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            print(f"[START] {request['anomaly_class']} q{request['query_id']} {request['scale']['name']} "
                  f"scalars {request['method']} {request['metric']}", flush=True)
            result = run_job(request)
            write_json(request["result_file"], result)
            print(f"[END {time.strftime('%Y-%m-%d %H:%M:%S %Z')}] AP={result['ap']:.6f} AUC={result['auc']:.6f} runtime={result['runtime_s']:.2f}s "
                  f"peak={result['peak_rss_mb']:.1f}MiB", flush=True)
            return 0
        except BaseException as exc:
            traceback.print_exc()
            result = {key: request[key] for key in ("anomaly_class", "method", "metric", "job_signature", "log_file")}
            result.update({"query": f"q{request['query_id']}", "query_id": request["query_id"],
                           "scale": request["scale"]["name"], "feature": "scalars", "status": "error",
                           "comparison_ready": False, "error": str(exc), "error_type": type(exc).__name__,
                           "traceback": traceback.format_exc(), "code_sha256": request["code_sha256"]})
            write_json(request["result_file"], result)
            return 1


def preflight():
    script = Path(__file__).resolve()
    code_hash = sha256_file(script)
    env_hash = digest_object(environment())
    pool_cache, jobs = {}, []
    for cls in ANOMALY_CLASSES:
        split_root = SPLIT_DIRS[cls]
        for query in QUERY_IDS:
            directory = INPUT_ROOT / f"Q{query}"
            normal = directory / f"q{query}_normal"
            anomaly = directory / f"q{query}_{'anomalies' if cls == 'S1' else 's2_anomalies'}"
            for pool in (normal, anomaly):
                if pool not in pool_cache:
                    files = source_files(pool)
                    with pq.ParquetFile(files[0]) as first:
                        missing = set(IDENTITY_COLUMNS + SCALAR_COLUMNS) - set(first.schema_arrow.names)
                    if missing:
                        raise ValueError(f"{files[0]}: missing {sorted(missing)}")
                    pool_cache[pool] = pool_fingerprint(pool, files)
            for scale in SCALES:
                manifest = split_path(split_root, cls, query, scale["name"])
                read_manifest(manifest, query, scale)
                manifest_hash = sha256_file(manifest)
                for method in UNSUP_METHODS:
                    for metric in DISTANCE_METRICS:
                        if method != "knn_distance" or metric != "cosine":
                            raise ValueError("Keep the original detector/metric settings or review the evaluation protocol")
                        stem = f"q{query}_{cls.lower()}_{scale['name']}_scalars_{method}_{metric}"
                        out = OUTPUT_ROOT / cls
                        request = {
                            "anomaly_class": cls, "query_id": query, "scale": scale,
                            "method": method, "metric": metric, "feature": "scalars",
                            "input_root": str(INPUT_ROOT), "normal_directory": str(normal),
                            "anomaly_directory": str(anomaly), "shared_split_file": str(manifest),
                            "split_sha256": manifest_hash, "code_sha256": code_hash,
                            "normal_input_fingerprint": pool_cache[normal],
                            "anomaly_input_fingerprint": pool_cache[anomaly],
                            "input_fingerprint": digest_object((pool_cache[normal], pool_cache[anomaly])),
                            "result_file": str(out / "WorkerResults" / (stem + ".json")),
                            "request_file": str(out / "Requests" / (stem + ".json")),
                            "log_file": str(out / "Logs" / (stem + ".log")),
                            "scores_file": str(out / "Scores" / (stem + "_scores.npz")),
                            "identity_audit_file": str(out / "SplitIdentityAudit" / (stem + ".json")),
                        }
                        request["job_signature"] = digest_object({
                            "class": cls, "query": query, "scale": scale, "method": method, "metric": metric,
                            "code": code_hash, "split": manifest_hash, "inputs": request["input_fingerprint"],
                            "environment": env_hash,
                        })
                        jobs.append(request)
    return jobs


def aggregate(jobs):
    results, errors = [], []
    for request in jobs:
        path = Path(request["result_file"])
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if row.get("job_signature") != request["job_signature"]:
            continue
        (results if row.get("status") == "ok" else errors).append(row)
    write_csv(OUTPUT_ROOT / "all_results.csv", results)
    write_csv(OUTPUT_ROOT / "all_errors.csv", errors)
    for cls in ANOMALY_CLASSES:
        write_csv(OUTPUT_ROOT / cls / f"scalars_{cls.lower()}_results_all.csv",
                  [r for r in results if r["anomaly_class"] == cls])
        write_csv(OUTPUT_ROOT / cls / f"scalars_{cls.lower()}_errors_all.csv",
                  [r for r in errors if r["anomaly_class"] == cls])



#!/usr/bin/env python3
"""Fixed Q28 witness-expansion sensitivity pilot, with portable paths."""
from __future__ import annotations


import csv


import gzip


import hashlib


import itertools


import json


import math


import platform


import resource


import shutil


import sys


import time


from datetime import datetime


from pathlib import Path


from typing import Any, Iterable, Sequence


import numpy as np


POOL_DIRS = {"normal": "q28_normal", "s1": "q28_anomalies", "s2": "q28_s2_anomalies"}


R_VALUES = (14, 15, 16)


BASELINE_R = 14


KMV_VALUES = (512, 1024)


VECTOR_DIMS = (512, 1024)


VECTOR_SKETCH_SEED = 13


EXACT_PRODUCT_LIMIT = 14**6


MAX_CANDIDATES = max(R_VALUES)**6


FACTOR_SAMPLE_SEED = b"q28-factor-sample-v1"


WITNESS_HASH_NAMESPACE = b"q28-vectorized-atomic-kmv-v1"


CODE_VERSION = "2026-11-04-q28-expansion-sensitivity-v1"


MASK64 = (1 << 64) - 1


SM64_GAMMA = 0x9E3779B97F4A7C15


SM64_MUL1 = 0xBF58476D1CE4E5B9


SM64_MUL2 = 0x94D049BB133111EB


PILOT_INPUTS = [
    {
        "pool": "normal",
        "role": "evaluation",
        "subtype": "normal",
        "query_name": "normal_06306"
    },
    {
        "pool": "normal",
        "role": "evaluation",
        "subtype": "normal",
        "query_name": "normal_41665"
    },
    {
        "pool": "normal",
        "role": "evaluation",
        "subtype": "normal",
        "query_name": "normal_50082"
    },
    {
        "pool": "normal",
        "role": "evaluation",
        "subtype": "normal",
        "query_name": "normal_58220"
    },
    {
        "pool": "normal",
        "role": "reference",
        "subtype": "normal",
        "query_name": "normal_05924"
    },
    {
        "pool": "normal",
        "role": "reference",
        "subtype": "normal",
        "query_name": "normal_12741"
    },
    {
        "pool": "normal",
        "role": "reference",
        "subtype": "normal",
        "query_name": "normal_22026"
    },
    {
        "pool": "normal",
        "role": "reference",
        "subtype": "normal",
        "query_name": "normal_32431"
    },
    {
        "pool": "normal",
        "role": "reference",
        "subtype": "normal",
        "query_name": "normal_44273"
    },
    {
        "pool": "normal",
        "role": "reference",
        "subtype": "normal",
        "query_name": "normal_47333"
    },
    {
        "pool": "normal",
        "role": "reference",
        "subtype": "normal",
        "query_name": "normal_48841"
    },
    {
        "pool": "normal",
        "role": "reference",
        "subtype": "normal",
        "query_name": "normal_53873"
    },
    {
        "pool": "s1",
        "role": "evaluation",
        "subtype": "s1_couponamt",
        "query_name": "s1_couponamt_00187"
    },
    {
        "pool": "s1",
        "role": "evaluation",
        "subtype": "s1_couponamt",
        "query_name": "s1_couponamt_00304"
    },
    {
        "pool": "s1",
        "role": "evaluation",
        "subtype": "s1_listprice",
        "query_name": "s1_listprice_00041"
    },
    {
        "pool": "s1",
        "role": "evaluation",
        "subtype": "s1_listprice",
        "query_name": "s1_listprice_00199"
    },
    {
        "pool": "s1",
        "role": "evaluation",
        "subtype": "s1_mix",
        "query_name": "s1_mix_00072"
    },
    {
        "pool": "s1",
        "role": "evaluation",
        "subtype": "s1_mix",
        "query_name": "s1_mix_00333"
    },
    {
        "pool": "s1",
        "role": "evaluation",
        "subtype": "s1_wholesalecost",
        "query_name": "s1_wholesalecost_00246"
    },
    {
        "pool": "s1",
        "role": "evaluation",
        "subtype": "s1_wholesalecost",
        "query_name": "s1_wholesalecost_00282"
    },
    {
        "pool": "s2",
        "role": "evaluation",
        "subtype": "blockremoveb6",
        "query_name": "blockremoveb6_00029"
    },
    {
        "pool": "s2",
        "role": "evaluation",
        "subtype": "blockremoveb6",
        "query_name": "blockremoveb6_00432"
    },
    {
        "pool": "s2",
        "role": "evaluation",
        "subtype": "droplistpricepredicate",
        "query_name": "droplistpricepredicate_00154"
    },
    {
        "pool": "s2",
        "role": "evaluation",
        "subtype": "droplistpricepredicate",
        "query_name": "droplistpricepredicate_00187"
    },
    {
        "pool": "s2",
        "role": "evaluation",
        "subtype": "dropwholesalecostpredicate",
        "query_name": "dropwholesalecostpredicate_00019"
    },
    {
        "pool": "s2",
        "role": "evaluation",
        "subtype": "dropwholesalecostpredicate",
        "query_name": "dropwholesalecostpredicate_00074"
    },
    {
        "pool": "s2",
        "role": "evaluation",
        "subtype": "joinadditem",
        "query_name": "joinadditem_00198"
    },
    {
        "pool": "s2",
        "role": "evaluation",
        "subtype": "joinadditem",
        "query_name": "joinadditem_00274"
    }
]


def parse_subtype(name: str) -> str:
    head, separator, tail = name.rpartition("_")
    return head if separator and tail.isdigit() else name


def compact_subtype(name: str) -> str:
    return parse_subtype(name).lower().replace("_", "")


def expected_factor_count(name: str) -> int:
    return 5 if "blockremoveb6" in compact_subtype(name) else 6


def expected_alternative_width(name: str) -> int:
    return 2 if "joinadditem" in compact_subtype(name) else 1


def is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def is_scalar_token(value: Any) -> bool:
    return value is not None and not is_sequence(value)


def parse_old_witness_string(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def flatten_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if is_scalar_token(value):
        return parse_old_witness_string(str(value))
    output: list[str] = []
    for element in value:
        output.extend(flatten_tokens(element))
    return output


def raw_factor_choices(component: Any) -> list[list[str]]:
    """Return the provenance alternatives represented by one Q28 factor."""
    if component is None:
        return []
    if is_scalar_token(component):
        return [[token] for token in parse_old_witness_string(str(component))]

    items = list(component)
    if not items:
        return []

    if all(is_scalar_token(item) for item in items):
        choices: list[list[str]] = []
        for item in items:
            choices.extend(
                [token] for token in parse_old_witness_string(str(item))
            )
        return choices

    return [flatten_tokens(alternative) for alternative in items]


def token_to_hash_i64(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def tuple_to_hash_i64(values: tuple[int, ...]) -> int:
    raw = b"".join(
        int(value).to_bytes(8, "big", signed=True) for value in values
    )
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def keyed_hash64_unsigned(value: int, seed: bytes) -> int:
    digest = hashlib.blake2b(
        str(int(value)).encode("utf-8"),
        digest_size=8,
        key=seed,
    ).digest()
    return int.from_bytes(digest, "big", signed=False)


def unsigned_to_signed_i64(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


def signed_to_unsigned_i64(value: int) -> int:
    return int(value) & ((1 << 64) - 1)


def canonical_alternative(tokens: Any) -> tuple[int, ...]:
    unique_tokens = {
        token.strip() for token in flatten_tokens(tokens) if token.strip()
    }
    return tuple(sorted(token_to_hash_i64(token) for token in unique_tokens))


def canonical_factor(component: Any, query_name: str, position: int) -> list[tuple[int, ...]]:
    alternatives = {
        canonical_alternative(choice) for choice in raw_factor_choices(component)
    }
    alternatives.discard(())
    if not alternatives:
        raise RuntimeError(
            f"{query_name}: provenance factor B{position} has no alternatives"
        )
    return sorted(alternatives)


def normalise_q28_factors(value: Any, query_name: str) -> list[list[tuple[int, ...]]]:
    if value is None or not is_sequence(value):
        raise RuntimeError(f"{query_name}: final prov is not a list of factors")

    factors = [
        canonical_factor(component, query_name, position)
        for position, component in enumerate(value, start=1)
    ]

    expected_count = expected_factor_count(query_name)
    if len(factors) != expected_count:
        raise RuntimeError(
            f"{query_name}: factor count={len(factors)}, expected={expected_count}"
        )

    expected_width = expected_alternative_width(query_name)
    widths = sorted({len(choice) for factor in factors for choice in factor})
    if widths != [expected_width]:
        raise RuntimeError(
            f"{query_name}: alternative widths={widths}, "
            f"expected={[expected_width]}"
        )
    return factors


def factor_sampling_rank(alternative: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    alternative_id = tuple_to_hash_i64(alternative)
    rank = keyed_hash64_unsigned(alternative_id, FACTOR_SAMPLE_SEED)
    return rank, alternative


def sampled_factors(
    factors: Sequence[Sequence[tuple[int, ...]]],
    max_r: int,
) -> list[list[tuple[int, ...]]]:
    return [
        sorted(factor, key=factor_sampling_rank)[:max_r]
        for factor in factors
    ]


def concrete_witness(selected: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    # Atomic Why uses set-valued witnesses.  Sorting the union makes the same
    # complete witness independent of factor traversal order.
    return tuple(sorted({token for alternative in selected for token in alternative}))


def namespace_salt(label: str) -> int:
    digest = hashlib.blake2b(
        WITNESS_HASH_NAMESPACE + b"\0" + label.encode("ascii"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def splitmix64_scalar(value: int) -> int:
    """Deterministic 64-bit mixer using Python integers."""
    value = (int(value) + SM64_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SM64_MUL1) & MASK64
    value = ((value ^ (value >> 27)) * SM64_MUL2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def splitmix64_array(values: np.ndarray | np.uint64) -> np.ndarray:
    """Vectorised counterpart of splitmix64_scalar."""
    array = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        array = array + np.uint64(SM64_GAMMA)
        array = (array ^ (array >> np.uint64(30))) * np.uint64(SM64_MUL1)
        array = (array ^ (array >> np.uint64(27))) * np.uint64(SM64_MUL2)
        return array ^ (array >> np.uint64(31))


def rotate_left_array(values: np.ndarray, shift: int) -> np.ndarray:
    shift = int(shift) % 64
    return (values << np.uint64(shift)) | (values >> np.uint64(64 - shift))


def complete_witness_rank_scalar(witness: Sequence[int]) -> int:
    """Hash one complete set-valued Atomic Why witness."""
    unique_tokens = {signed_to_unsigned_i64(token) for token in witness}
    sum_part = 0
    xor_part = 0
    for token in unique_tokens:
        sum_part = (
            sum_part + splitmix64_scalar(token ^ TOKEN_SUM_SALT)
        ) & MASK64
        xor_part ^= splitmix64_scalar(token ^ TOKEN_XOR_SALT)
    rotated = ((xor_part << 23) | (xor_part >> 41)) & MASK64
    arity_part = splitmix64_scalar(len(unique_tokens) ^ ARITY_SALT)
    return splitmix64_scalar(sum_part ^ rotated ^ arity_part ^ FINAL_SALT)


def finalise_witness_ranks(
    sum_part: np.ndarray,
    xor_part: np.ndarray,
    arity: np.ndarray | int,
) -> np.ndarray:
    arity_values = np.asarray(arity, dtype=np.uint64)
    arity_part = splitmix64_array(arity_values ^ np.uint64(ARITY_SALT))
    combined = (
        sum_part
        ^ rotate_left_array(xor_part, 23)
        ^ arity_part
        ^ np.uint64(FINAL_SALT)
    )
    return splitmix64_array(combined)


def factor_domains_are_disjoint(
    factors: Sequence[Sequence[tuple[int, ...]]],
) -> bool:
    seen: set[int] = set()
    for factor in factors:
        domain = {
            token
            for alternative in factor
            for token in alternative
        }
        if seen.intersection(domain):
            return False
        seen.update(domain)
    return True


def disjoint_witness_ranks(
    factors: Sequence[Sequence[tuple[int, ...]]],
) -> np.ndarray:
    """Hash a Cartesian product when factor token domains do not overlap."""
    shape = tuple(len(factor) for factor in factors)
    sum_state = np.zeros(shape, dtype=np.uint64)
    xor_state = np.zeros(shape, dtype=np.uint64)
    total_arity = 0

    for axis, factor in enumerate(factors):
        widths = {len(alternative) for alternative in factor}
        if len(widths) != 1:
            raise RuntimeError("A sampled Q28 factor has mixed alternative widths")
        width = widths.pop()
        total_arity += width
        tokens = np.asarray(factor, dtype=np.int64).view(np.uint64)
        sum_signature = np.sum(
            splitmix64_array(tokens ^ np.uint64(TOKEN_SUM_SALT)),
            axis=1,
            dtype=np.uint64,
        )
        xor_signature = np.bitwise_xor.reduce(
            splitmix64_array(tokens ^ np.uint64(TOKEN_XOR_SALT)),
            axis=1,
        )
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = len(factor)
        with np.errstate(over="ignore"):
            sum_state += sum_signature.reshape(broadcast_shape)
        xor_state ^= xor_signature.reshape(broadcast_shape)

    return finalise_witness_ranks(sum_state, xor_state, total_arity).reshape(-1)


def overlapping_witness_ranks(
    factors: Sequence[Sequence[tuple[int, ...]]],
    chunk_size: int = 200_000,
) -> np.ndarray:
    """Hash set-valued witnesses exactly when factor token domains overlap."""
    dimensions = tuple(len(factor) for factor in factors)
    factor_arrays = [np.asarray(factor, dtype=np.int64) for factor in factors]
    total = math.prod(dimensions)
    ranks = np.empty(total, dtype=np.uint64)

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        flat_indices = np.arange(start, stop, dtype=np.int64)
        coordinates = np.unravel_index(flat_indices, dimensions)
        token_matrix = np.concatenate(
            [array[coordinates[axis]] for axis, array in enumerate(factor_arrays)],
            axis=1,
        )
        token_matrix.sort(axis=1)
        unique_mask = np.ones(token_matrix.shape, dtype=bool)
        if token_matrix.shape[1] > 1:
            unique_mask[:, 1:] = token_matrix[:, 1:] != token_matrix[:, :-1]

        token_bits = token_matrix.view(np.uint64)
        sum_terms = splitmix64_array(token_bits ^ np.uint64(TOKEN_SUM_SALT))
        xor_terms = splitmix64_array(token_bits ^ np.uint64(TOKEN_XOR_SALT))
        sum_terms[~unique_mask] = np.uint64(0)
        xor_terms[~unique_mask] = np.uint64(0)
        sum_part = np.sum(sum_terms, axis=1, dtype=np.uint64)
        xor_part = np.bitwise_xor.reduce(xor_terms, axis=1)
        arity = np.sum(unique_mask, axis=1, dtype=np.uint64)
        ranks[start:stop] = finalise_witness_ranks(sum_part, xor_part, arity)

    return ranks


def vectorised_witness_ranks(
    factors: Sequence[Sequence[tuple[int, ...]]],
) -> np.ndarray:
    if any(not factor for factor in factors):
        raise RuntimeError("At least one sampled Q28 factor is empty")
    if factor_domains_are_disjoint(factors):
        return disjoint_witness_ranks(factors)
    return overlapping_witness_ranks(factors)


def bottom_k_distinct(values: np.ndarray, k: int) -> list[int]:
    """Return the k smallest distinct unsigned ranks without sorting all values."""
    flat = np.asarray(values, dtype=np.uint64).reshape(-1)
    if flat.size == 0:
        return []

    take = min(flat.size, max(int(k) * 4, int(k)))
    while True:
        if take == flat.size:
            candidates = flat
        else:
            candidates = np.partition(flat, take - 1)[:take]
        unique = np.unique(candidates)
        if len(unique) >= k or take == flat.size:
            ordered = np.sort(unique)[:k]
            return [unsigned_to_signed_i64(int(value)) for value in ordered]
        take = min(flat.size, take * 2)


def kmv_jaccard(left: Sequence[int], right: Sequence[int], k: int) -> float:
    left_set = {signed_to_unsigned_i64(value) for value in left}
    right_set = {signed_to_unsigned_i64(value) for value in right}
    union_bottom = sorted(left_set | right_set)[:k]
    if not union_bottom:
        return 1.0
    shared = sum(
        value in left_set and value in right_set for value in union_bottom
    )
    return shared / len(union_bottom)


def provenance_column_index(columns: Sequence[str]) -> int:
    lowered = [str(column).lower() for column in columns]
    matches = [index for index, column in enumerate(lowered) if column == "prov"]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one final prov column, found {len(matches)}: {columns}"
        )
    return matches[0]


def vector_sketch_salts(family: str) -> tuple[int, int, int, int]:
    prefix = f"prov-cs-v2-{family}-{VECTOR_SKETCH_SEED}"
    output: list[int] = []
    for suffix in (
        "token-bucket",
        "token-sign",
        "arity-bucket",
        "arity-sign",
    ):
        digest = hashlib.blake2b(
            f"{prefix}-{suffix}".encode("ascii"),
            digest_size=8,
        ).digest()
        output.append(int.from_bytes(digest, "big", signed=False))
    return output[0], output[1], output[2], output[3]


def atomic_witness_countsketch(
    witness_ranks: np.ndarray,
    dim: int,
) -> np.ndarray:
    """CountSketch binary complete-witness identifiers into a dense vector."""
    ranks = np.asarray(witness_ranks, dtype=np.uint64).reshape(-1)
    bucket_salt, sign_salt, _, _ = vector_sketch_salts("atomic-witness")
    buckets = splitmix64_array(ranks ^ np.uint64(bucket_salt)) % np.uint64(dim)
    positive = splitmix64_array(ranks ^ np.uint64(sign_salt)) & np.uint64(1)
    signs = np.where(positive == 1, 1.0, -1.0)
    output = np.bincount(
        buckets.astype(np.int64),
        weights=signs,
        minlength=dim,
    )
    return output.astype(np.float32)


def factor_alternatives_have_private_identifiers(
    factors: Sequence[Sequence[tuple[int, ...]]],
) -> bool:
    """Prove that different factor choices yield different witnesses.

    Every alternative must contain a factor-private token signature, and those
    signatures must be unique within that factor.  This holds for Q28 because
    every alternative contains its own fact-table provenance token, including
    the join-add-item subtype where dimension tokens may repeat across factors.
    """
    factor_domains = [
        {token for alternative in factor for token in alternative}
        for factor in factors
    ]
    for index, factor in enumerate(factors):
        other_tokens: set[int] = set()
        for other_index, domain in enumerate(factor_domains):
            if other_index != index:
                other_tokens.update(domain)
        signatures = [
            tuple(sorted(set(alternative).difference(other_tokens)))
            for alternative in factor
        ]
        if any(not signature for signature in signatures):
            return False
        if len(set(signatures)) != len(signatures):
            return False
    return True


TOKEN_SUM_SALT = namespace_salt("token-sum")


TOKEN_XOR_SALT = namespace_salt("token-xor")


ARITY_SALT = namespace_salt("arity")


FINAL_SALT = namespace_salt("final")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def array_digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def factors_digest(factors):
    digest = hashlib.sha256()
    for factor in factors:
        digest.update(len(factor).to_bytes(8, "big"))
        for alternative in factor:
            digest.update(len(alternative).to_bytes(8, "big"))
            for token in alternative:
                digest.update(int(token).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


def build_summary(factors):
    count = math.prod(len(factor) for factor in factors)
    if count > MAX_CANDIDATES:
        raise RuntimeError(f"Refusing expansion of {count:,} candidates; limit={MAX_CANDIDATES:,}")
    if not factor_alternatives_have_private_identifiers(factors):
        raise RuntimeError("Distinct factor choices cannot be proved to give distinct witnesses. No result produced.")
    start = time.perf_counter()
    ranks = vectorised_witness_ranks(factors)
    if len(ranks) != count:
        raise RuntimeError("Cartesian candidate count mismatch")
    hashing_seconds = time.perf_counter() - start
    start = time.perf_counter()
    ordered = bottom_k_distinct(ranks, max(KMV_VALUES))
    if len(ordered) != min(count, max(KMV_VALUES)):
        raise RuntimeError("Unexpected number of distinct KMV hashes")
    kmv_seconds = time.perf_counter() - start
    vectors, timings = {}, {}
    for dim in VECTOR_DIMS:
        start = time.perf_counter()
        vectors[dim] = atomic_witness_countsketch(ranks, dim)
        timings[dim] = time.perf_counter() - start
        if not np.isfinite(vectors[dim]).all():
            raise RuntimeError("Non-finite CountSketch")
    return {
        "kmv": {k: ordered[:k] for k in KMV_VALUES},
        "vectors": vectors,
        "candidate_count": count,
        "hashing_seconds": hashing_seconds,
        "kmv_seconds": kmv_seconds,
        "countsketch_seconds": timings,
    }


def self_test():
    examples = [
        [[(100 * axis + j,) for j in range(3)] for axis in range(6)],
        [[(1, 2), (1, 3)], [(1, 4), (1, 5)], [(6,), (7,)]],
    ]
    for factors in examples:
        witnesses = {concrete_witness(c) for c in itertools.product(*factors)}
        expected = np.asarray(sorted(complete_witness_rank_scalar(w) for w in witnesses), dtype=np.uint64)
        actual = vectorised_witness_ranks(factors)
        if not np.array_equal(np.sort(actual), expected):
            raise AssertionError("Vectorised witness hashes differ from explicit set expansion")
        if len(witnesses) != math.prod(map(len, factors)):
            raise AssertionError("Synthetic injective product is not injective")
        if not factor_alternatives_have_private_identifiers(factors):
            raise AssertionError("Injectivity check rejected a valid example")
        summary = build_summary(factors)
        for k in KMV_VALUES:
            wanted = [unsigned_to_signed_i64(int(v)) for v in expected[:k]]
            if summary["kmv"][k] != wanted:
                raise AssertionError("KMV does not match exact synthetic expansion")
        bucket_salt, sign_salt, _, _ = vector_sketch_salts("atomic-witness")
        for dim in VECTOR_DIMS:
            wanted = np.zeros(dim, dtype=np.float32)
            for rank in expected:
                bucket = splitmix64_scalar(int(rank) ^ bucket_salt) % dim
                sign = 1 if splitmix64_scalar(int(rank) ^ sign_salt) & 1 else -1
                wanted[bucket] += sign
            if not np.array_equal(summary["vectors"][dim], wanted):
                raise AssertionError("CountSketch differs from explicit witness sketch")
    factors = [[(1000 * axis + j,) for j in range(20)] for axis in range(6)]
    ranked = sampled_factors(factors, max(R_VALUES))
    for small, large in zip(R_VALUES, R_VALUES[1:]):
        if not all(set(f[:small]).issubset(f[:large]) for f in ranked):
            raise AssertionError("Factor selections are not nested")
    if factor_alternatives_have_private_identifiers([[(1,), (2,)], [(1,), (2,)]]):
        raise AssertionError("Non-injective product was not rejected")
    if kmv_jaccard([1, 2], [2, 3], 512) != 1 / 3:
        raise AssertionError("KMV Jaccard self-test failed")
    print("[SELF-TEST] exact synthetic expansion, set semantics, KMV and CountSketch: PASS", flush=True)


def cosine_similarity(left, right):
    # Float32 L2 normalisation, as used for the stored vector evaluation.
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))


def describe(values, prefix):
    array = np.asarray(values, dtype=np.float64)
    return {
        prefix + "_median": float(np.median(array)),
        prefix + "_p95": float(np.quantile(array, 0.95)),
        prefix + "_max": float(np.max(array)),
    }


def make_reports(output, items, results, costs):
    references = [item for item in items if item["role"] == "reference" and item["key"] in results]
    evaluations = [item for item in items if item["role"] == "evaluation" and item["key"] in results]
    if not references or not evaluations:
        return []
    refs = sorted(item["key"] for item in references)
    methods = [("kmv", k) for k in KMV_VALUES] + [("countsketch", d) for d in VECTOR_DIMS]
    scores, nearest, index = [], [], {}
    for r in R_VALUES:
        for family, size in methods:
            for item in evaluations:
                key = item["key"]
                distances = {}
                for ref in refs:
                    a, b = results[key][r], results[ref][r]
                    similarity = (kmv_jaccard(a["kmv"][size], b["kmv"][size], size)
                                  if family == "kmv" else cosine_similarity(a["vectors"][size], b["vectors"][size]))
                    distance = 1.0 - similarity
                    distances[ref] = distance
                    scores.append({"r": r, "family": family, "size": size,
                                   "pool": item["pool"], "subtype": item["subtype"],
                                   "query_name": item["query_name"], "reference": ref,
                                   "similarity": similarity, "distance": distance})
                best = min(distances.values())
                ties = {ref for ref in refs if distances[ref] == best}
                index[r, family, size, key] = distances, best, ties
                nearest.append({"r": r, "family": family, "size": size,
                                "pool": item["pool"], "subtype": item["subtype"],
                                "query_name": item["query_name"], "score": best,
                                "nearest_reference": min(ties), "tied_references": len(ties)})
    write_csv(output / "pairwise_scores.csv", scores)
    write_csv(output / "nearest_scores.csv", nearest)
    comparisons = []
    for low, high in itertools.combinations(R_VALUES, 2):
        for family, size in methods:
            for group in ("all", "normal", "s1", "s2"):
                group_items = [item for item in evaluations if group == "all" or item["pool"] == group]
                if not group_items:
                    continue
                pair_delta, score_delta = [], []
                strict, tie_aware = 0, 0
                for item in group_items:
                    a, sa, ta = index[low, family, size, item["key"]]
                    b, sb, tb = index[high, family, size, item["key"]]
                    pair_delta.extend(abs(a[ref] - b[ref]) for ref in refs)
                    score_delta.append(abs(sa - sb))
                    strict += min(ta) == min(tb)
                    tie_aware += bool(ta & tb)
                row = {"from_r": low, "to_r": high, "family": family, "size": size,
                       "group": group, "n_evaluations": len(group_items), "n_references": len(refs),
                       "n_pairs": len(pair_delta), **describe(pair_delta, "pair_delta"),
                       **describe(score_delta, "nearest_score_delta"),
                       "top1_agreement": strict / len(group_items),
                       "tie_aware_top1_agreement": tie_aware / len(group_items)}
                # Preserve, rather than relax, the former KMV acceptance rule.
                row["original_kmv_criteria_pass"] = (
                    row["pair_delta_median"] <= 0.010 and row["pair_delta_p95"] <= 0.020
                    and row["top1_agreement"] >= 0.950) if family == "kmv" else "not_defined"
                comparisons.append(row)
    write_csv(output / "sensitivity.csv", comparisons)
    cost_summary = []
    for r in R_VALUES:
        subset = [row for row in costs if row["r"] == r]
        if not subset:
            continue
        cost_summary.append({"r": r, "executions": len(subset),
                             "mean_candidates": sum(row["candidates"] for row in subset) / len(subset),
                             "total_construction_seconds": sum(row["construction_seconds"] for row in subset),
                             "mean_construction_seconds": sum(row["construction_seconds"] for row in subset) / len(subset)})
    write_csv(output / "cost_summary.csv", cost_summary)
    return comparisons


def run_pilot():
    import duckdb

    self_test()
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}. Check --database; no database will be created.")
    items = []
    for original in PILOT_INPUTS:
        item = dict(original)
        item["key"] = item["pool"] + ":" + item["query_name"]
        item["path"] = DATA_ROOT / POOL_DIRS[item["pool"]] / (item["query_name"] + ".sql")
        if not item["path"].is_file():
            raise FileNotFoundError(f"Fixed pilot input missing: {item['path']}. No substitute selected.")
        item["sql"] = item["path"].read_text(encoding="utf-8")
        item["sql_sha256"] = hashlib.sha256(item["sql"].encode("utf-8")).hexdigest()
        items.append(item)
    if len({item["key"] for item in items}) != len(items):
        raise RuntimeError("Duplicate pilot input")

    started = time.perf_counter()
    output = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "sketches").mkdir()
    (output / "selected_factors").mkdir()
    db_stat = DB_PATH.stat()
    run = {
        "status": "RUNNING", "version": CODE_VERSION,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "database": str(DB_PATH), "database_bytes": db_stat.st_size, "database_mtime_ns": db_stat.st_mtime_ns,
        "data_root": str(DATA_ROOT), "r_values": R_VALUES, "kmv_sizes": KMV_VALUES,
        "countsketch_dimensions": VECTOR_DIMS, "vector_seed": VECTOR_SKETCH_SEED,
        "factor_seed": FACTOR_SAMPLE_SEED.decode(), "witness_hash_namespace": WITNESS_HASH_NAMESPACE.decode(),
        "exact_product_limit": EXACT_PRODUCT_LIMIT, "largest_reduced_product": MAX_CANDIDATES,
        "reference_count": sum(i["role"] == "reference" for i in items),
        "evaluation_count": sum(i["role"] == "evaluation" for i in items),
        "r_order": "rotated across executions to reduce fixed ordering effects",
        "measurement_scope": "Construction times exclude SQL, factor preparation and output writes; RSS is the cumulative process peak, not per-configuration memory.",
        "interpretation": "Follow-up sensitivity study on a fixed pilot. Largest r is not full-population ground truth. No automatic optimum or budget validation is claimed.",
        "coverage": "Atomic KMV and binary Atomic CountSketch only; no frequency-weighted bag validation, AP or threshold calibration.",
        "python": sys.version, "numpy": np.__version__, "duckdb": duckdb.__version__,
        "platform": platform.platform(),
    }
    write_json(output / "run.json", run)
    write_csv(output / "inputs.csv", [{k: str(v) for k, v in item.items() if k != "sql"} for item in items])
    print(f"[START] {len(items)} fixed SQLs; r={R_VALUES}; Atomic KMV and binary CountSketch", flush=True)
    print(f"[INPUT] {DATA_ROOT}\n[DB] {DB_PATH}\n[OUTPUT] {output}", flush=True)
    print("Existing SQL, splits and Parquets are read-only. No dataset is regenerated.", flush=True)
    costs, executions, errors, results = [], [], [], {}
    interrupted = False
    try:
        with duckdb.connect(str(DB_PATH), read_only=True) as connection:
            connection.execute("SET schema = ?", [DB_SCHEMA])
            connection.execute(f"SET threads = {DUCKDB_THREADS}")
            connection.execute("SET memory_limit = '4GB'")
            for position, item in enumerate(items):
                name, key = item["query_name"], item["key"]
                print(f"[SQL {position+1}/{len(items)}] {item['pool']} {name}", flush=True)
                try:
                    statements = connection.extract_statements(item["sql"])
                    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                        raise RuntimeError("Pilot accepts exactly one SELECT per SQL file")
                    tick = time.perf_counter()
                    cursor = connection.execute(item["sql"])
                    column = provenance_column_index([d[0] for d in cursor.description])
                    rows = cursor.fetchall()
                    sql_seconds = time.perf_counter() - tick
                    if len(rows) != 1:
                        raise RuntimeError(f"Expected one Q28 result row, received {len(rows)}")
                    tick = time.perf_counter()
                    factors = normalise_q28_factors(rows[0][column], name)
                    del rows
                    factor_sizes = list(map(len, factors))
                    full_candidates = math.prod(factor_sizes)
                    ranked = sampled_factors(factors, max(R_VALUES))
                    lineage = sorted({t for f in factors for alt in f for t in alt})
                    mode = "exact" if full_candidates <= EXACT_PRODUCT_LIMIT else "reduced"
                    saved = factors if mode == "exact" else ranked
                    with gzip.open(output / "selected_factors" / (item["pool"] + "_" + name + ".json.gz"), "wt", encoding="utf-8") as handle:
                        json.dump({"mode": mode, "original_factor_sizes": factor_sizes,
                                   "selected_up_to_r": max(R_VALUES), "factors": saved,
                                   "sql_sha256": item["sql_sha256"]}, handle)
                    executions.append({"pool": item["pool"], "role": item["role"], "query_name": name,
                                       "factor_sizes": json.dumps(factor_sizes), "full_product_candidates": str(full_candidates),
                                       "mode": mode, "full_factor_sha256": factors_digest(factors),
                                       "lineage_distinct_tokens": len(lineage), "lineage_sha256": array_digest(lineage),
                                       "sql_seconds": sql_seconds, "factor_preparation_seconds": time.perf_counter() - tick})
                    built = {}
                    shift = position % len(R_VALUES)
                    order = R_VALUES[shift:] + R_VALUES[:shift]
                    for r in order:
                        chosen = factors if mode == "exact" else [factor[:r] for factor in ranked]
                        candidates = math.prod(map(len, chosen))
                        print(f"  [BUILD] r={r}; candidates={candidates:,}", flush=True)
                        tick = time.perf_counter()
                        summary = build_summary(chosen)
                        elapsed = time.perf_counter() - tick
                        built[r] = summary
                        arrays = {f"atomic_kmv_{k}": np.asarray(summary["kmv"][k], dtype=np.int64) for k in KMV_VALUES}
                        arrays.update({f"atomic_countsketch_{d}": summary["vectors"][d] for d in VECTOR_DIMS})
                        np.savez_compressed(output / "sketches" / f"{item['pool']}_{name}_r{r}.npz", **arrays)
                        costs.append({"pool": item["pool"], "query_name": name, "r": r, "mode": mode,
                                      "candidates": candidates, "construction_seconds": elapsed,
                                      "hashing_seconds": summary["hashing_seconds"], "kmv_both_sizes_seconds": summary["kmv_seconds"],
                                      **{f"countsketch_{d}_seconds": summary["countsketch_seconds"][d] for d in VECTOR_DIMS},
                                      "process_cumulative_peak_rss_mib": peak_rss_mib(),
                                      **{f"countsketch_{d}_zero": not bool(np.any(summary["vectors"][d])) for d in VECTOR_DIMS}})
                        write_csv(output / "construction.csv", costs)
                        print(f"  [DONE] r={r}; construction={elapsed:.2f}s; run_elapsed={time.perf_counter()-started:.1f}s", flush=True)
                    results[key] = built
                except Exception as exc:
                    errors.append({"pool": item["pool"], "query_name": name, "error_type": type(exc).__name__, "error": str(exc)})
                    print(f"[FAILED] {name}: {type(exc).__name__}: {exc}", flush=True)
                    write_csv(output / "errors.csv", errors)
                write_csv(output / "executions.csv", executions)
    except KeyboardInterrupt:
        interrupted = True
        print("[INTERRUPTED] Keeping partial output; no source files changed.", flush=True)
    except Exception as exc:
        errors.append({"pool": "", "query_name": "", "error_type": type(exc).__name__, "error": str(exc)})
        print(f"[FAILED] {type(exc).__name__}: {exc}", flush=True)

    complete = not interrupted and not errors and len(results) == len(items)
    # Incomplete runs do not silently change the reference or evaluation sample.
    comparisons = make_reports(output, items, results, costs) if complete else []
    write_csv(output / "errors.csv", errors)
    run.update(status="COMPLETE" if complete else "INCOMPLETE", completed_queries=len(results),
               errors=len(errors), interrupted=interrupted, elapsed_seconds=time.perf_counter()-started,
               process_peak_rss_mib=peak_rss_mib())
    write_json(output / "run.json", run)
    for row in comparisons:
        if row["group"] == "all" and row["from_r"] == BASELINE_R:
            print(f"[SENSITIVITY] {row['family']}-{row['size']} r{row['from_r']} -> r{row['to_r']}: "
                  f"score_delta_p95={row['nearest_score_delta_p95']:.6f}; "
                  f"score_delta_max={row['nearest_score_delta_max']:.6f}; top1={row['top1_agreement']:.1%}", flush=True)
    (output / "README.txt").write_text(
        "This is a follow-up expansion-reduction sensitivity test, not a rerun of the article evaluation.\n"
        "sensitivity.csv: absolute pairwise/nearest-score changes and neighbour agreement by group.\n"
        "Original KMV acceptance thresholds remain reported; CountSketch has no invented acceptance rule.\n"
        "The largest r is another approximation, not full Cartesian ground truth. No automatic selection is made.\n"
        "No AP, frequency-weighted bag, threshold-classification or full-workload validation is claimed.\n"
        "construction.csv: construction costs. RSS is cumulative for this process, not per-r.\n"
        "SQL is executed once per selected query; selected factors and sketches are saved.\n"
        "A run is usable as a complete pilot only when run.json status is COMPLETE.\n", encoding="utf-8")
    archive = shutil.make_archive(str(output), "zip", root_dir=output)
    print(f"[{run['status']}] {len(results)}/{len(items)} queries; ZIP: {archive}", flush=True)
    return 0 if complete else 1



def main(argv=None):
    import argparse
    global DATA_ROOT, DB_PATH, DB_SCHEMA, OUTPUT_ROOT, DUCKDB_THREADS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path)
    parser.add_argument('--queries-root', type=Path, help='The Q28 folder containing q28_normal, q28_anomalies and q28_s2_anomalies')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--schema', default='tpcds')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if any(v is None for v in (args.database, args.queries_root, args.output_dir)):
        parser.error('--database, --queries-root and --output-dir are required')
    DB_PATH, DATA_ROOT, OUTPUT_ROOT = (p.resolve() for p in (args.database, args.queries_root, args.output_dir))
    DB_SCHEMA, DUCKDB_THREADS = args.schema, args.threads
    if args.threads < 1 or not args.schema.strip():
        parser.error('Positive threads and a nonempty schema are required')
    if not DB_PATH.is_file() or not DATA_ROOT.is_dir():
        parser.error('Existing database and Q28 SQL directory are required')
    if DATA_ROOT == OUTPUT_ROOT or DATA_ROOT in OUTPUT_ROOT.parents or OUTPUT_ROOT in DATA_ROOT.parents or OUTPUT_ROOT == DB_PATH:
        parser.error('Output must be separate from input data')
    return run_pilot()


if __name__ == '__main__':
    raise SystemExit(main())

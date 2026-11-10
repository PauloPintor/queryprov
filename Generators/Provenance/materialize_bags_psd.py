#!/usr/bin/env python3
"""Materialise frequency CountSketch and PSD sidecars from existing SQL and compact batches.

Use --help for paths and execution options. See README.md for output and resume policies.
"""

from __future__ import annotations

import ast
import csv
import concurrent.futures
import contextlib
import fcntl
import hashlib
import heapq
import itertools
import json
import math
import multiprocessing
import os
import re
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# Paths and query selection are supplied through the command line.
# =============================================================================
SERVER_ROOT = None
INPUT_ROOT = None
# Expected location of the SAME annotated database used on Arch.
# Change only this line if the database was copied elsewhere on Contabo.
DB_PATH = None
DB_SCHEMA = "tpcds"
#QUERY_IDS = (3, 5, 18, 20, 28, 60, 66, 82, 84, 85, 89, 92, 93, 94, 98)
QUERY_IDS = ()
DATASETS = ("normal", "s1", "s2")
POOL_SUFFIX = {"normal": "normal", "s1": "anomalies", "s2": "s2_anomalies"}
PARQUET_BATCH_SIZE = 25
FETCH_ROWS = 64
DUCKDB_THREADS = 2
MAX_WORKERS = 4
DUCKDB_MEMORY_PER_WORKER = "8GB"
COMPRESSION = "zstd"
SIDECAR_SUFFIX = ".bagcompact.parquet"
MAX_WSIZE_BIN = 8
BAG_PSD_VERSION = "2026-10-01-weighted-cs-psd-v1"
IDENTITY_COLUMNS = ("query_id", "template_id", "query_name", "subtype")
PSD_COUNT_COLUMNS = ("nrows", "nwitness_unique", "nwitness_total", "ntokens_unique", "ntokens_total")
PSD_FEATURE_NAMES = (*PSD_COUNT_COLUMNS, "token_unique_ratio", "witness_unique_ratio", "witness_token_ratio")
BASELINE_COLUMNS = tuple(f"{f}_kmv_{k}" for f in ("lineage", "monoids") for k in (512, 1024)) + tuple(
    f"{f}_countsketch_{k}" for f in ("lineage", "atomic") for k in (512, 1024))

KMV_KS = (512, 1024)
MAX_KMV_K = max(KMV_KS)
VECTOR_DIMS = (512, 1024)
VECTOR_SKETCH_SEED = 13
KMV_SEED = b"prov-kmv-v1"
FACTOR_SAMPLE_R = 14
EXACT_PRODUCT_LIMIT = FACTOR_SAMPLE_R**6
FACTOR_SAMPLE_SEED = b"q28-factor-sample-v1"
WITNESS_HASH_NAMESPACE = b"q28-vectorized-atomic-kmv-v1"
CODE_VERSION = "2026-09-27-all-queries-compact-v1"
# Hash/schema contract above is unchanged; execution implementation is versioned separately.
EXECUTION_VERSION = "2026-09-27-parallel-sql-v2"

# These pools use an outer SUM of products of alternative witnesses.
FACTORIZED_QUERY_IDS = frozenset({92, 94})


# =============================================================================
# Existing Q28 r14 implementation, retained without changing its hash policy.
# =============================================================================

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
    """Return the alternatives represented by one Q28 product factor."""
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


def canonical_factor(
    component: Any,
    query_name: str,
    position: int,
) -> list[tuple[int, ...]]:
    alternatives = {
        canonical_alternative(choice) for choice in raw_factor_choices(component)
    }
    alternatives.discard(())
    if not alternatives:
        raise RuntimeError(
            f"{query_name}: provenance factor B{position} has no alternatives"
        )
    return sorted(alternatives)


def normalise_q28_factors(
    value: Any,
    query_name: str,
) -> list[list[tuple[int, ...]]]:
    if value is None or not is_sequence(value):
        raise RuntimeError(f"{query_name}: final prov is not a list of factors")

    factors = [
        canonical_factor(component, query_name, position)
        for position, component in enumerate(value, start=1)
    ]

    expected_count = expected_factor_count(query_name)
    if len(factors) != expected_count:
        raise RuntimeError(
            f"{query_name}: factor count={len(factors)}, "
            f"expected={expected_count}"
        )

    expected_width = expected_alternative_width(query_name)
    widths = sorted(
        {len(choice) for factor in factors for choice in factor}
    )
    if widths != [expected_width]:
        raise RuntimeError(
            f"{query_name}: alternative widths={widths}, "
            f"expected={[expected_width]}"
        )
    return factors


def factor_sampling_rank(
    alternative: tuple[int, ...],
) -> tuple[int, tuple[int, ...]]:
    alternative_id = tuple_to_hash_i64(alternative)
    rank = keyed_hash64_unsigned(alternative_id, FACTOR_SAMPLE_SEED)
    return rank, alternative


def bottom_r_factor(
    factor: Sequence[tuple[int, ...]],
    r: int,
) -> list[tuple[int, ...]]:
    return heapq.nsmallest(r, factor, key=factor_sampling_rank)


def concrete_witness(
    selected: Sequence[tuple[int, ...]],
) -> tuple[int, ...]:
    # Atomic Why witnesses are sets.  Sorting the union gives a canonical
    # identity independent of factor traversal order.
    return tuple(
        sorted({token for alternative in selected for token in alternative})
    )


class BottomK:
    """Retain the k smallest distinct unsigned 64-bit values."""

    def __init__(self, k: int) -> None:
        self.k = int(k)
        self.heap: list[int] = []
        self.values: set[int] = set()

    def add(self, value: int) -> None:
        value = int(value)
        if value in self.values:
            return
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, -value)
            self.values.add(value)
            return

        current_maximum = -self.heap[0]
        if value >= current_maximum:
            return

        removed = -heapq.heapreplace(self.heap, -value)
        self.values.remove(removed)
        self.values.add(value)

    def ordered_signed(self) -> list[int]:
        return [
            unsigned_to_signed_i64(value) for value in sorted(self.values)
        ]


def kmv_from_identifiers(identifiers: Iterable[int], k: int) -> list[int]:
    keeper = BottomK(k)
    for identifier in identifiers:
        keeper.add(keyed_hash64_unsigned(int(identifier), KMV_SEED))
    return keeper.ordered_signed()


U64_MASK = (1 << 64) - 1
MIX64_C1 = 0xBF58476D1CE4E5B9
MIX64_C2 = 0x94D049BB133111EB


def mix_u64_scalar(value: int, salt: int) -> int:
    x = ((int(value) & U64_MASK) ^ int(salt)) & U64_MASK
    x ^= x >> 30
    x = (x * MIX64_C1) & U64_MASK
    x ^= x >> 27
    x = (x * MIX64_C2) & U64_MASK
    x ^= x >> 31
    return x & U64_MASK


def mix_u64_array(values: np.ndarray, salt: int) -> np.ndarray:
    signed = np.asarray(values, dtype=np.int64)
    mixed = signed.view(np.uint64).copy()
    mixed ^= np.uint64(salt)
    with np.errstate(over="ignore"):
        mixed ^= mixed >> np.uint64(30)
        mixed *= np.uint64(MIX64_C1)
        mixed ^= mixed >> np.uint64(27)
        mixed *= np.uint64(MIX64_C2)
        mixed ^= mixed >> np.uint64(31)
    return mixed


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


def lineage_countsketch(
    identifiers: Iterable[int],
    dim: int,
) -> np.ndarray:
    """CountSketch a binary Lineage set into a fixed-dimensional vector."""
    values = np.asarray(sorted(set(identifiers)), dtype=np.int64)
    output = np.zeros(dim, dtype=np.float64)
    if len(values) == 0:
        return output.astype(np.float32)

    bucket_salt, sign_salt, _, _ = vector_sketch_salts("lineage")
    buckets = mix_u64_array(values, bucket_salt) % np.uint64(dim)
    positive_bits = mix_u64_array(values, sign_salt) & np.uint64(1)
    signs = np.where(positive_bits == 1, 1.0, -1.0)
    np.add.at(output, buckets.astype(np.int64), signs)
    return output.astype(np.float32)


SM64_GAMMA = 0x9E3779B97F4A7C15


def namespace_salt(label: str) -> int:
    digest = hashlib.blake2b(
        WITNESS_HASH_NAMESPACE + b"\0" + label.encode("ascii"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


TOKEN_SUM_SALT = namespace_salt("token-sum")
TOKEN_XOR_SALT = namespace_salt("token-xor")
ARITY_SALT = namespace_salt("arity")
FINAL_SALT = namespace_salt("final")


def splitmix64_scalar(value: int) -> int:
    value = (int(value) + SM64_GAMMA) & U64_MASK
    value = ((value ^ (value >> 30)) * MIX64_C1) & U64_MASK
    value = ((value ^ (value >> 27)) * MIX64_C2) & U64_MASK
    return (value ^ (value >> 31)) & U64_MASK


def splitmix64_array(values: np.ndarray | np.uint64) -> np.ndarray:
    array = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        array = array + np.uint64(SM64_GAMMA)
        array = (array ^ (array >> np.uint64(30))) * np.uint64(MIX64_C1)
        array = (array ^ (array >> np.uint64(27))) * np.uint64(MIX64_C2)
        return array ^ (array >> np.uint64(31))


def rotate_left_array(values: np.ndarray, shift: int) -> np.ndarray:
    shift = int(shift) % 64
    return (values << np.uint64(shift)) | (values >> np.uint64(64 - shift))


def complete_witness_rank_scalar(witness: Sequence[int]) -> int:
    unique_tokens = {signed_to_unsigned_i64(token) for token in witness}
    sum_part = 0
    xor_part = 0
    for token in unique_tokens:
        sum_part = (
            sum_part + splitmix64_scalar(token ^ TOKEN_SUM_SALT)
        ) & U64_MASK
        xor_part ^= splitmix64_scalar(token ^ TOKEN_XOR_SALT)
    rotated = ((xor_part << 23) | (xor_part >> 41)) & U64_MASK
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


def disjoint_witness_ranks(
    factors: Sequence[Sequence[tuple[int, ...]]],
) -> np.ndarray:
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
        raise RuntimeError("At least one selected Q28 factor is empty")
    if factor_token_domains_are_disjoint(factors):
        return disjoint_witness_ranks(factors)
    return overlapping_witness_ranks(factors)


def bottom_k_distinct(values: np.ndarray, k: int) -> list[int]:
    flat = np.asarray(values, dtype=np.uint64).reshape(-1)
    if flat.size == 0:
        return []

    take = min(flat.size, max(int(k) * 4, int(k)))
    while True:
        candidates = (
            flat
            if take == flat.size
            else np.partition(flat, take - 1)[:take]
        )
        unique = np.unique(candidates)
        if len(unique) >= k or take == flat.size:
            ordered = np.sort(unique)[:k]
            return [unsigned_to_signed_i64(int(value)) for value in ordered]
        take = min(flat.size, take * 2)


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


def factor_token_domains_are_disjoint(
    factors: Sequence[Sequence[tuple[int, ...]]],
) -> bool:
    observed: set[int] = set()
    for factor in factors:
        factor_tokens = {
            token for alternative in factor for token in alternative
        }
        if observed.intersection(factor_tokens):
            return False
        observed.update(factor_tokens)
    return True


def exact_lineage_identifiers(
    factors: Sequence[Sequence[tuple[int, ...]]],
) -> set[int]:
    return {
        token
        for factor in factors
        for alternative in factor
        for token in alternative
    }


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


def atomic_summaries(
    factors: list[list[tuple[int, ...]]],
) -> tuple[list[int], dict[int, np.ndarray], str, int, int | None, bool]:
    """Build Atomic KMV and vectors from one shared witness population."""
    exact_count = math.prod(len(factor) for factor in factors)
    if exact_count <= EXACT_PRODUCT_LIMIT:
        selected_factors = factors
        mode = "exact_expansion"
        sample_r: int | None = None
    else:
        selected_factors = [
            bottom_r_factor(factor, FACTOR_SAMPLE_R) for factor in factors
        ]
        mode = "bounded_factor_expansion"
        sample_r = FACTOR_SAMPLE_R

    candidate_count = math.prod(len(factor) for factor in selected_factors)
    injective = factor_alternatives_have_private_identifiers(selected_factors)
    if not injective:
        raise RuntimeError(
            "Q28 factor alternatives do not provide unique private identifiers; "
            "binary Atomic CountSketch would require explicit deduplication"
        )

    witness_ranks = vectorised_witness_ranks(selected_factors)
    if len(witness_ranks) != candidate_count:
        raise RuntimeError(
            f"Generated {len(witness_ranks)} witness hashes, "
            f"expected {candidate_count}"
        )

    sketch = bottom_k_distinct(witness_ranks, MAX_KMV_K)
    expected_maximum = min(MAX_KMV_K, candidate_count)
    if len(sketch) != expected_maximum:
        raise RuntimeError(
            f"Invalid Atomic Why KMV length {len(sketch)} for "
            f"{candidate_count} candidates"
        )
    vectors = {
        dim: atomic_witness_countsketch(witness_ranks, dim)
        for dim in VECTOR_DIMS
    }
    return sketch, vectors, mode, candidate_count, sample_r, injective


# =============================================================================
# ORDINARY AND SMALL FACTORISED QUERY EXECUTIONS
# =============================================================================


def flat_tokens(value: Any) -> list[str]:
    """A concrete witness cannot contain another list of witnesses."""
    if value is None:
        return []
    parts = value if is_sequence(value) else [value]
    tokens: list[str] = []
    for part in parts:
        if part is None:
            continue
        if is_sequence(part):
            raise ValueError("Unexpected nested provenance; refusing to hash a list as a token")
        tokens.extend(parse_old_witness_string(str(part)))
    return tokens


def alternatives(component: Any) -> list[list[str]]:
    if component is None or component == []:
        # Preserve the existing Q92/Q94 convention for absent optional factors.
        return [[]]
    if not is_sequence(component):
        return [[token] for token in flat_tokens(component)] or [[]]
    if all(not is_sequence(item) for item in component):
        return [[token] for token in flat_tokens(component)] or [[]]
    if any(item is not None and not is_sequence(item) for item in component):
        raise ValueError("Mixed scalar/list alternatives in a factor")
    return [flat_tokens(item) for item in component if item is not None] or [[]]


def expanded_cell(cell: Any, query: int) -> Iterable[list[str]]:
    if cell is None:
        return
    outer = cell if is_sequence(cell) else [cell]
    for term in outer:
        if term is None:
            continue
        nested = is_sequence(term) and any(is_sequence(x) for x in term)
        if not nested:
            yield flat_tokens(term)
            continue
        if query not in FACTORIZED_QUERY_IDS:
            raise ValueError(
                f"Q{query}: unexpected nested provenance. For Q5, use the corrected "
                "SQL with list concatenation at the final aggregate; do not guess SUM/PRODUCT."
            )
        factors = [alternatives(part) for part in term]
        count = math.prod(map(len, factors))
        if count > EXACT_PRODUCT_LIMIT:
            raise ValueError(f"Q{query}: product has {count:,} alternatives; no silent truncation")
        for selected in itertools.product(*factors):
            yield [token for choice in selected for token in choice]


def provenance_column(columns: Sequence[str]) -> int:
    matches = [i for i, name in enumerate(columns)
               if name.lower() in {"witnesses", "prov", "token"}]
    if not matches:
        raise ValueError(f"No provenance column: {list(columns)}")
    if len(matches) > 1 and matches[-1] != len(columns) - 1:
        raise ValueError(f"Ambiguous provenance columns: {list(columns)}")
    return matches[-1]


def weighted_countsketch(identifiers: Sequence[int], weights: Sequence[int],
                         dim: int, family: str) -> list[float]:
    if len(identifiers) != len(weights):
        raise ValueError("Invalid positive frequency weights")
    if not len(identifiers):
        return [0.0] * dim
    bucket_salt, sign_salt, _, _ = vector_sketch_salts(
        "lineage" if family == "lineage" else "atomic-witness")
    if family == "lineage":
        values = np.asarray(identifiers, dtype=np.int64)
        buckets = mix_u64_array(values, bucket_salt) % np.uint64(dim)
        positive = mix_u64_array(values, sign_salt) & np.uint64(1)
    elif family == "atomic":
        values = np.asarray(identifiers, dtype=np.uint64)
        buckets = splitmix64_array(values ^ np.uint64(bucket_salt)) % np.uint64(dim)
        positive = splitmix64_array(values ^ np.uint64(sign_salt)) & np.uint64(1)
    else:
        raise ValueError(f"Unknown sketch family: {family}")
    weights_array = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights_array).all() or np.any(weights_array <= 0):
        raise OverflowError("Bag frequencies cannot be represented as float64")
    output = np.bincount(buckets.astype(np.int64),
                         weights=weights_array * np.where(positive == 1, 1.0, -1.0),
                         minlength=dim)
    if not np.isfinite(output).all():
        raise OverflowError("Bag sketch overflow")
    return output.tolist()


def size_histogram(counter: Counter) -> list[int]:
    histogram = [0] * (MAX_WSIZE_BIN + 1)
    for size, count in counter.items():
        if size > 0:
            histogram[min(size, MAX_WSIZE_BIN + 1) - 1] += count
    return histogram


def set_partitions(size: int) -> list[tuple[int, ...]]:
    partitions = [()]
    for position in range(size):
        bit = 1 << position
        partitions = [new for partition in partitions for new in (
            [partition + (bit,)] +
            [partition[:i] + (block | bit,) + partition[i + 1:]
             for i, block in enumerate(partition)])]
    return sorted(partitions, key=lambda p: (len(p), p))


def q28_size_counts(factors: Sequence[Counter]) -> Counter:
    """Exact size histogram for Q28 singleton factors or its added-item join.

    The join case has one private fact token plus one item token per choice.
    Equality patterns of at most six item positions (203 partitions) replace
    enumeration of the witness product. No sampling is used for this histogram.
    """
    choices = [list(f) for f in factors]
    total = math.prod(sum(f.values()) for f in factors)
    if factor_token_domains_are_disjoint(choices):
        return Counter({sum(len(f[0]) for f in choices): total})
    if any(len(w) != 2 for f in choices for w in f):
        raise ValueError("Unsupported Q28 shared-token pattern for exact size histogram")
    domains = [set(itertools.chain.from_iterable(f)) for f in choices]
    item_counts = []
    for i, factor in enumerate(factors):
        others = set().union(*(d for j, d in enumerate(domains) if j != i))
        occurrences = Counter(t for w in factor for t in w)
        items = Counter()
        for witness, frequency in factor.items():
            private = [t for t in witness if t not in others and occurrences[t] == 1]
            if not private:
                raise ValueError("Q28 needs a unique private fact token per alternative")
            fact_token = min(private)
            item = next(t for t in witness if t != fact_token)
            items[item] += frequency
        item_counts.append(items)
    moments = {}
    for mask in range(1, 1 << len(factors)):
        groups = [item_counts[i] for i in range(len(factors)) if mask & (1 << i)]
        smallest = min(groups, key=len)
        moments[mask] = sum(math.prod(g.get(t, 0) for g in groups) for t in smallest)
    exact_patterns = {}
    sizes = Counter()
    for partition in set_partitions(len(factors)):
        count = math.prod(moments[block] for block in partition)
        for coarser, already_counted in exact_patterns.items():
            if len(coarser) < len(partition) and all(
                    any((block & big) == block for big in coarser) for block in partition):
                count -= already_counted
        if count < 0:
            raise RuntimeError("Negative Q28 histogram count")
        exact_patterns[partition] = count
        sizes[len(factors) + len(partition)] += count
    if sum(sizes.values()) != total:
        raise RuntimeError("Q28 histogram does not cover the full witness population")
    return sizes


def q28_full_statistics(factors: Sequence[Counter]) -> tuple[Counter, int, int, Counter]:
    choices = [list(f) for f in factors]
    if not factor_alternatives_have_private_identifiers(choices):
        raise ValueError("Cannot prove unique Q28 witnesses; exact distinct PSD count unsupported")
    totals = [sum(f.values()) for f in factors]
    total = math.prod(totals)
    unique = math.prod(map(len, factors))
    occurrences: dict[int, dict[int, int]] = {}
    for i, factor in enumerate(factors):
        for alternative, frequency in factor.items():
            for token in alternative:
                per_factor = occurrences.setdefault(token, {})
                per_factor[i] = per_factor.get(i, 0) + frequency
    # Count witnesses containing a token via the complement (none of the factors
    # selected it). This also handles an item token shared by several factors.
    token_counts = Counter({
        token: total - math.prod(totals[i] - per_factor.get(i, 0) for i in range(len(factors)))
        for token, per_factor in occurrences.items()
    })
    sizes = q28_size_counts(factors)
    if sum(token_counts.values()) != sum(size * count for size, count in sizes.items()):
        raise RuntimeError("Q28 token counts and size histogram disagree")
    return token_counts, unique, total, sizes


def psd_fields(nrows: int, witness_unique: int, witness_total: int,
               token_counts: Counter, sizes: Counter) -> dict[str, Any]:
    counts = (nrows, witness_unique, witness_total, len(token_counts), sum(token_counts.values()))
    n, wu, wt, tu, tt = counts
    if min(counts) < 0 or wu > wt or tu > tt:
        raise ValueError("Invalid PSD counts")
    if sum(sizes.values()) != wt or sum(k * v for k, v in sizes.items()) != tt:
        raise ValueError("PSD histogram and total counts disagree")
    histogram = size_histogram(sizes)
    fields = {key: float(value) for key, value in zip(PSD_COUNT_COLUMNS, counts)}
    fields.update({key + "_decimal": str(value) for key, value in zip(PSD_COUNT_COLUMNS, counts)})
    fields.update({
        "psd_features": [float(v) for v in counts] + [tu / max(1, tt), wu / max(1, wt), wu / max(1, tu)],
        "avg_w": tt / wt if wt else 0.0,
        "wsize_hist": [float(v) for v in histogram],
        "wsize_hist_decimal": [str(v) for v in histogram],
        "psd_population": "full", "psd_counts_exact": True,
        "psd_feature_names": list(PSD_FEATURE_NAMES),
    })
    return fields


def ordinary_bag_summary(cursor: Any, column: int, query: int) -> tuple[dict, dict]:
    token_counts: Counter = Counter()
    witnesses: Counter = Counter()
    sizes: Counter = Counter()
    tokens_seen: dict[str, int] = {}
    nrows = 0
    while rows := cursor.fetchmany(FETCH_ROWS):
        nrows += len(rows)
        for row in rows:
            for raw in expanded_cell(row[column], query):
                ids = set()
                for token in raw:
                    token = token.strip()
                    if token:
                        if token not in tokens_seen:
                            tokens_seen[token] = token_to_hash_i64(token)
                        ids.add(tokens_seen[token])
                if not ids:
                    continue
                witness = tuple(sorted(ids))
                witnesses[witness] += 1
                token_counts.update(witness)
                sizes[len(witness)] += 1
                if len(witnesses) > EXACT_PRODUCT_LIMIT:
                    raise ValueError("Too many distinct witnesses; no unannounced sampling")
    ordered = sorted(witnesses)
    ranks = [complete_witness_rank_scalar(w) for w in ordered]
    atomic_ids = [tuple_to_hash_i64(w) for w in ordered]
    if len(set(ranks)) != len(ordered) or len(set(atomic_ids)) != len(ordered):
        raise RuntimeError("Atomic hash collision detected")
    result = psd_fields(nrows, len(witnesses), sum(witnesses.values()), token_counts, sizes)
    baseline = {}
    lineage_kmv = kmv_from_identifiers(token_counts, MAX_KMV_K)
    atomic_kmv = kmv_from_identifiers(atomic_ids, MAX_KMV_K)
    for k in KMV_KS:
        baseline[f"lineage_kmv_{k}"] = lineage_kmv[:k]
        baseline[f"monoids_kmv_{k}"] = atomic_kmv[:k]
    lineage_ids = sorted(token_counts)
    for dim in VECTOR_DIMS:
        result[f"bag_lineage_countsketch_{dim}"] = weighted_countsketch(
            lineage_ids, [token_counts[t] for t in lineage_ids], dim, "lineage")
        result[f"bag_atomic_countsketch_{dim}"] = weighted_countsketch(
            ranks, [witnesses[w] for w in ordered], dim, "atomic")
        baseline[f"lineage_countsketch_{dim}"] = lineage_countsketch(lineage_ids, dim).tolist()
        baseline[f"atomic_countsketch_{dim}"] = atomic_witness_countsketch(np.asarray(ranks, dtype=np.uint64), dim).tolist()
    result.update({"bag_lineage_population": "full", "bag_atomic_population": "full",
                   "bag_atomic_selected_unique_decimal": str(len(witnesses)),
                   "bag_atomic_selected_total_decimal": str(sum(witnesses.values())),
                   "factor_sample_r": None, "factor_sizes": [],
                   "psd_method": "streamed_full_witness_occurrences"})
    return result, baseline


def q28_bag_summary(cursor: Any, column: int, name: str) -> tuple[dict, dict]:
    rows = cursor.fetchmany(2)
    if len(rows) != 1:
        raise ValueError("Q28 must return one aggregate row")
    value = rows[0][column]
    canonical = normalise_q28_factors(value, name)
    factors = [Counter(canonical_alternative(w) for w in raw_factor_choices(component))
               for component in value]
    for counter, expected in zip(factors, canonical):
        counter.pop((), None)
        if set(counter) != set(expected):
            raise ValueError("Q28 bag and binary alternatives disagree")
    token_counts, full_unique, full_total, sizes = q28_full_statistics(factors)
    bounded = full_unique > EXACT_PRODUCT_LIMIT
    selected = [bottom_r_factor(f, FACTOR_SAMPLE_R) for f in canonical] if bounded else canonical
    candidate_count = math.prod(map(len, selected))
    ranks = vectorised_witness_ranks(selected)
    if len(ranks) != candidate_count:
        raise RuntimeError("Incorrect Q28 selected witness count")
    # Product weights count all occurrences of each selected complete witness.
    selected_total = math.prod(sum(factors[i][w] for w in factor) for i, factor in enumerate(selected))
    unit_weights = all(factors[i][w] == 1 for i, factor in enumerate(selected) for w in factor)
    if unit_weights:
        weights = np.ones(candidate_count, dtype=np.float64)
    else:
        weights = np.ones(tuple(map(len, selected)), dtype=np.float64)
        for axis, factor in enumerate(selected):
            shape = [1] * len(selected)
            shape[axis] = len(factor)
            weights *= np.asarray([factors[axis][w] for w in factor], dtype=np.float64).reshape(shape)
        weights = weights.reshape(-1)
    result = psd_fields(1, full_unique, full_total, token_counts, sizes)
    baseline = {}
    lineage_ids = sorted(token_counts)
    lineage_kmv = kmv_from_identifiers(lineage_ids, MAX_KMV_K)
    atomic_kmv = bottom_k_distinct(ranks, MAX_KMV_K)
    for k in KMV_KS:
        baseline[f"lineage_kmv_{k}"] = lineage_kmv[:k]
        baseline[f"monoids_kmv_{k}"] = atomic_kmv[:k]
    for dim in VECTOR_DIMS:
        result[f"bag_lineage_countsketch_{dim}"] = weighted_countsketch(
            lineage_ids, [token_counts[t] for t in lineage_ids], dim, "lineage")
        result[f"bag_atomic_countsketch_{dim}"] = weighted_countsketch(ranks, weights, dim, "atomic")
        baseline[f"lineage_countsketch_{dim}"] = lineage_countsketch(lineage_ids, dim).tolist()
        baseline[f"atomic_countsketch_{dim}"] = (
            np.asarray(result[f"bag_atomic_countsketch_{dim}"], dtype=np.float32).tolist()
            if unit_weights else atomic_witness_countsketch(ranks, dim).tolist())
    result.update({"bag_lineage_population": "full",
                   "bag_atomic_population": "bounded_factor_expansion" if bounded else "full",
                   "bag_atomic_selected_unique_decimal": str(candidate_count),
                   "bag_atomic_selected_total_decimal": str(selected_total),
                   "factor_sample_r": FACTOR_SAMPLE_R if bounded else None,
                   "factor_sizes": list(map(len, canonical)),
                   "psd_method": "exact_factor_arithmetic"})
    return result, baseline


def summarise_bags(cursor: Any, query: int, name: str) -> tuple[dict, dict]:
    column = provenance_column([d[0] for d in cursor.description])
    result, baseline = (q28_bag_summary(cursor, column, name) if query == 28
                        else ordinary_bag_summary(cursor, column, query))
    result.update({"bag_psd_version": BAG_PSD_VERSION, "vector_sketch_seed": VECTOR_SKETCH_SEED,
                   "bag_weights": "witness_occurrence_frequency",
                   "witness_semantics": "set_of_distinct_tokens",
                   "bag_vectors_normalised": False,
                   "lineage_sketch_algorithm": "prov-cs-v2-lineage",
                   "atomic_sketch_algorithm": "atomic-witness-countsketch-splitmix64-v1"})
    validate_bag_record(result)
    return result, baseline


def validate_bag_record(record: dict) -> None:
    counts = [int(record[key + "_decimal"]) for key in PSD_COUNT_COLUMNS]
    n, wu, wt, tu, tt = counts
    if min(counts) < 0 or wu > wt or tu > tt:
        raise ValueError("Invalid exact PSD counts")
    expected = [float(v) for v in counts] + [tu / max(1, tt), wu / max(1, wt), wu / max(1, tu)]
    if record["psd_features"] != expected:
        raise ValueError("PSD feature vector disagrees with exact counts")
    for key, value in zip(PSD_COUNT_COLUMNS, counts):
        if record[key] != float(value):
            raise ValueError(f"PSD numeric/decimal mismatch: {key}")
    histogram = [int(v) for v in record["wsize_hist_decimal"]]
    if len(histogram) != MAX_WSIZE_BIN + 1 or min(histogram) < 0 or sum(histogram) != wt:
        raise ValueError("Invalid full witness-size histogram")
    if record["wsize_hist"] != [float(v) for v in histogram]:
        raise ValueError("Histogram numeric/decimal mismatch")
    if not np.isfinite(expected + [record["avg_w"]] + record["wsize_hist"]).all():
        raise ValueError("Non-finite PSD values")
    if record["avg_w"] != (tt / wt if wt else 0.0):
        raise ValueError("Incorrect mean witness size")
    selected = int(record["bag_atomic_selected_unique_decimal"])
    selected_total = int(record["bag_atomic_selected_total_decimal"])
    if not (0 <= selected <= wu and selected <= selected_total <= wt):
        raise ValueError("Invalid selected population counts")
    if record["bag_atomic_population"] == "full" and (selected, selected_total) != (wu, wt):
        raise ValueError("Full Atomic bag has incomplete population")
    for family in ("lineage", "atomic"):
        for dim in VECTOR_DIMS:
            v = np.asarray(record[f"bag_{family}_countsketch_{dim}"])
            if v.shape != (dim,) or not np.isfinite(v).all():
                raise ValueError(f"Invalid {family} bag sketch {dim}")
    if record["bag_vectors_normalised"] or record["psd_population"] != "full":
        raise ValueError("Unexpected normalisation/PSD population")
    if (record["psd_feature_names"] != list(PSD_FEATURE_NAMES) or not record["psd_counts_exact"]
            or record["bag_lineage_population"] != "full"
            or record["bag_atomic_population"] not in {"full", "bounded_factor_expansion"}):
        raise ValueError("Unexpected feature order or population metadata")
    if record["bag_psd_version"] != BAG_PSD_VERSION or record["vector_sketch_seed"] != VECTOR_SKETCH_SEED:
        raise ValueError("Unexpected bag version or hash seed")


def bag_schema() -> pa.Schema:
    fields = [("query_id", pa.int64()), ("template_id", pa.int64()),
              ("query_name", pa.string()), ("subtype", pa.string()),
              ("query", pa.string()), ("pool", pa.string()), ("sql_sha256", pa.string())]
    fields.extend((k, pa.float64()) for k in PSD_COUNT_COLUMNS)
    fields.extend((k + "_decimal", pa.string()) for k in PSD_COUNT_COLUMNS)
    fields.extend((f"bag_{family}_countsketch_{dim}", pa.list_(pa.float64()))
                  for family in ("lineage", "atomic") for dim in VECTOR_DIMS)
    fields.extend((k, pa.string()) for k in (
        "psd_population", "psd_method", "bag_lineage_population", "bag_atomic_population",
        "bag_atomic_selected_unique_decimal", "bag_atomic_selected_total_decimal",
        "bag_psd_version", "bag_weights", "witness_semantics",
        "lineage_sketch_algorithm", "atomic_sketch_algorithm"))
    fields.extend([
        ("psd_features", pa.list_(pa.float64())), ("psd_feature_names", pa.list_(pa.string())),
        ("psd_counts_exact", pa.bool_()), ("avg_w", pa.float64()),
        ("wsize_hist", pa.list_(pa.float64())), ("wsize_hist_decimal", pa.list_(pa.string())),
        ("factor_sample_r", pa.int16()), ("factor_sizes", pa.list_(pa.int64())),
        ("vector_sketch_seed", pa.int16()), ("bag_vectors_normalised", pa.bool_()),
        ("sql_execute_seconds", pa.float64()), ("summarisation_seconds", pa.float64()),
        ("baseline_columns_verified", pa.list_(pa.string())),
    ])
    return pa.schema(fields)


def digest_object(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_sql_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"SQL is not UTF-8: {path}. Nothing was changed.") from exc


def read_main_batch(path: Path, query: int) -> list[dict]:
    with pq.ParquetFile(path) as parquet:
        columns = parquet.schema_arrow.names
        if not {"query_id", "query_name"}.issubset(columns):
            raise ValueError(f"No execution identities in {path}")
        read_columns = [k for k in (*IDENTITY_COLUMNS, "sql_sha256", *BASELINE_COLUMNS) if k in columns]
        records = parquet.read(columns=read_columns).to_pylist()
    if not records:
        raise ValueError(f"Empty main Parquet: {path}")
    for row in records:
        name = row["query_name"]
        if not isinstance(name, str) or not name or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"Unsafe query name in {path}")
        row["query_id"] = int(row["query_id"])
        row["template_id"] = int(row.get("template_id") if row.get("template_id") is not None else query)
        row["subtype"] = str(row.get("subtype") or parse_subtype(name))
    return records


def database_fingerprint() -> dict:
    stat = DB_PATH.stat()
    return {"path": str(DB_PATH.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def computation_fingerprint() -> str:
    # Query/pool selection, paths and worker count can change without changing
    # the summaries. Function bodies and mathematical constants cannot.
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    definitions = [ast.dump(node, include_attributes=False) for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    return digest_object({"definitions": definitions, "version": BAG_PSD_VERSION,
                          "dims": VECTOR_DIMS, "seed": VECTOR_SKETCH_SEED,
                          "kmv_seed": KMV_SEED.hex(), "factor_r": FACTOR_SAMPLE_R,
                          "factor_seed": FACTOR_SAMPLE_SEED.hex(),
                          "namespace": WITNESS_HASH_NAMESPACE.hex(),
                          "exact_limit": EXACT_PRODUCT_LIMIT, "histogram_max": MAX_WSIZE_BIN,
                          "psd_order": PSD_FEATURE_NAMES,
                          "factorised_queries": sorted(FACTORIZED_QUERY_IDS),
                          "mix": [U64_MASK, MIX64_C1, MIX64_C2, SM64_GAMMA,
                                  TOKEN_SUM_SALT, TOKEN_XOR_SALT, ARITY_SALT, FINAL_SALT]})


def execute_bag_task(connection: Any, task: tuple) -> dict:
    query, pool, identity, sql, sql_hash = task
    started = time.perf_counter()
    cursor = connection.execute(sql)
    sql_seconds = time.perf_counter() - started
    summary_start = time.perf_counter()
    record, baseline = summarise_bags(cursor, query, identity["query_name"])
    verified = []
    for key in BASELINE_COLUMNS:
        if key in identity and identity[key] is not None:
            if identity[key] != baseline[key]:
                raise ValueError(f"{identity['query_name']}: recomputed {key} differs from the existing "
                                 "main Parquet; sidecar not written. Check SQL/database/hash version.")
            verified.append(key)
    record.update({k: identity[k] for k in IDENTITY_COLUMNS})
    record.update({"query": f"q{query}", "pool": pool, "sql_sha256": sql_hash,
                   "sql_execute_seconds": sql_seconds,
                   "summarisation_seconds": time.perf_counter() - summary_start,
                   "baseline_columns_verified": verified})
    return record


_BAG_DB = None


def initialise_bag_worker(db_path: str, scratch: str, schema: str, threads: int, memory: str) -> None:
    global DB_SCHEMA, DUCKDB_THREADS, DUCKDB_MEMORY_PER_WORKER
    DB_SCHEMA, DUCKDB_THREADS, DUCKDB_MEMORY_PER_WORKER = schema, threads, memory
    global _BAG_DB
    _BAG_DB = duckdb.connect(db_path, read_only=True, config={
        "threads": DUCKDB_THREADS, "memory_limit": DUCKDB_MEMORY_PER_WORKER,
        "temp_directory": str(Path(scratch) / str(os.getpid())),
    })
    _BAG_DB.execute("SET schema = '" + DB_SCHEMA.replace("'", "''") + "'")


def bag_worker(task: tuple) -> dict:
    if _BAG_DB is None:
        raise RuntimeError("Worker not initialised")
    return execute_bag_task(_BAG_DB, task)


def verify_sidecar(path: Path, digest: str, expected: list[dict]) -> list[dict] | None:
    if not path.exists():
        return None
    with pq.ParquetFile(path) as parquet:
        metadata = parquet.schema_arrow.metadata or {}
        if metadata.get(b"bag_psd_input_sha256") != digest.encode():
            raise ValueError(f"Existing sidecar belongs to another format/input: {path}. "
                             "It was not replaced; preserve or move it before a new run.")
        if not parquet.schema_arrow.equals(bag_schema(), check_metadata=False):
            raise ValueError(f"Unexpected sidecar schema: {path}")
        table = parquet.read()
        table.validate(full=True)
        records = table.to_pylist()
    if len(records) != len(expected):
        raise ValueError(f"Incomplete sidecar: {path}")
    for record, identity in zip(records, expected):
        if any(record[k] != identity[k] for k in (*IDENTITY_COLUMNS, "sql_sha256")):
            raise ValueError(f"Sidecar identities/SQL disagree: {path}")
        validate_bag_record(record)
    return records


def commit_sidecar(path: Path, records: list[dict], digest: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing sidecar: {path}")
    schema = bag_schema().with_metadata({b"bag_psd_input_sha256": digest.encode(),
                                       b"bag_psd_version": BAG_PSD_VERSION.encode()})
    for record in records:
        validate_bag_record(record)
    fd, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        pq.write_table(pa.Table.from_pylist(records, schema=schema), temporary, compression=COMPRESSION)
        verify_sidecar(temporary, digest, records)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Hard-link publishes a complete file without overwriting any existing one.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_bag_pool(connection: Any, executor: Any, query: int, pool: str,
                     db_stamp: dict, code_hash: str) -> None:
    directory = INPUT_ROOT / f"Q{query}" / f"q{query}_{POOL_SUFFIX[pool]}"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    if (directory / "_run.json").exists() and not (directory / "_SUCCESS.json").exists():
        raise ValueError(f"Compact materialisation is incomplete: {directory}")
    files = sorted(p for p in directory.iterdir() if re.fullmatch(r"batch_[0-9]+\.parquet", p.name))
    if not files:
        raise ValueError(f"No existing main Parquets with identities: {directory}")
    sidecar_names = {p.stem + SIDECAR_SUFFIX for p in files}
    unexpected = {p.name for p in directory.glob("batch_*" + SIDECAR_SUFFIX)} - sidecar_names
    if unexpected:
        raise ValueError(f"Unmatched sidecars in {directory}: {sorted(unexpected)}")
    marker = directory / "_BAG_PSD_SUCCESS.json"
    marker.unlink(missing_ok=True)
    completed = 0
    seen_names: set[str] = set()
    seen_ids: set[int] = set()
    batch_digests = {}
    started = time.perf_counter()
    print(f"\n[Q{query} {pool}] main batches={len(files)}; output={directory}", flush=True)
    for index, source in enumerate(files, 1):
        identities = read_main_batch(source, query)
        original_main_digest = digest_object(identities)
        tasks = []
        for identity in identities:
            if identity["query_name"] in seen_names or identity["query_id"] in seen_ids:
                raise ValueError(f"Repeated execution identity in {source}")
            seen_names.add(identity["query_name"])
            seen_ids.add(identity["query_id"])
            sql_path = directory / (identity["query_name"] + ".sql")
            sql = read_sql_file(sql_path)
            sql_hash = hashlib.sha256(sql.encode()).hexdigest()
            if identity.get("sql_sha256") and identity["sql_sha256"] != sql_hash:
                raise ValueError(f"SQL changed since main Parquet: {sql_path}")
            identity["sql_sha256"] = sql_hash
            tasks.append((query, pool, identity, sql, sql_hash))
        digest = digest_object({"version": BAG_PSD_VERSION, "code": code_hash, "db": db_stamp,
                                "schema": DB_SCHEMA, "query": query, "pool": pool,
                                "dims": VECTOR_DIMS, "seed": VECTOR_SKETCH_SEED,
                                "factor_r": FACTOR_SAMPLE_R, "identities_and_baseline": identities})
        destination = directory / (source.stem + SIDECAR_SUFFIX)
        if database_fingerprint() != db_stamp:
            raise RuntimeError("Database changed during the run; pool incomplete")
        cached = verify_sidecar(destination, digest, identities)
        if cached is not None:
            status = "RESUME"
        else:
            print(f"[BUILD {index}/{len(files)}] {source.name}; executions={len(tasks)}", flush=True)
            records = ([execute_bag_task(connection, t) for t in tasks] if executor is None
                       else list(executor.map(bag_worker, tasks, chunksize=1)))
            if database_fingerprint() != db_stamp:
                raise RuntimeError("Database changed; new batch not committed")
            # Re-read the main identity/baseline to detect a concurrent replacement.
            fresh = read_main_batch(source, query)
            if digest_object(fresh) != original_main_digest:
                raise RuntimeError(f"Main Parquet changed during the run: {source}")
            commit_sidecar(destination, records, digest)
            status = "WRITE"
        completed += len(identities)
        batch_digests[source.name] = digest
        print(f"[{status} {index}/{len(files)}] {destination.name}; "
              f"executions={completed:,}; elapsed={time.perf_counter() - started:.1f}s", flush=True)
    sql_names = {p.stem for p in directory.glob("*.sql") if not p.name.startswith(".")}
    if sql_names != seen_names:
        raise ValueError(f"SQL/Parquet coverage mismatch in {directory}: "
                         f"SQL-only={len(sql_names - seen_names)}, Parquet-only={len(seen_names - sql_names)}")
    atomic_json(marker, {"bag_psd_version": BAG_PSD_VERSION, "query": query, "pool": pool,
                         "rows": completed, "batches": batch_digests, "db": db_stamp,
                         "psd_feature_names": PSD_FEATURE_NAMES,
                         "normalisation": "none; use L2 in the evaluator"})


def run_bag_self_test() -> None:
    class TestCursor:
        description = [("prov",)]
        def __init__(self, rows):
            self.rows = list(rows)
        def fetchmany(self, n):
            first, self.rows = self.rows[:n], self.rows[n:]
            return first
    a, _ = summarise_bags(TestCursor([([["a", "b"]],)]), 3, "test")
    b, _ = summarise_bags(TestCursor([([["a", "b"], ["a", "b"]],)]), 3, "test")
    for family in ("lineage", "atomic"):
        for dim in VECTOR_DIMS:
            np.testing.assert_array_equal(np.asarray(a[f"bag_{family}_countsketch_{dim}"]) * 2,
                                          b[f"bag_{family}_countsketch_{dim}"])
    factors = [Counter({(i * 10,): 2, (i * 10 + 1,): 1}) for i in range(6)]
    tc, wu, wt, sizes = q28_full_statistics(factors)
    assert wu == 2**6 and wt == 3**6 and sum(tc.values()) == 6 * 3**6
    assert sizes == Counter({6: 3**6})
    for query, cell, explicit in (
        (92, [[[["a", "b", "c"]], [["b", "d"], ["e", "f"]]]],
         [["a", "b", "c", "d"], ["a", "b", "c", "e", "f"]]),
        (94, [[["ws"], ["d"], ["ca"], ["site"], ["n1", "n1", "n2"]]],
         [["ws", "d", "ca", "site", t] for t in ("n1", "n1", "n2")]),
    ):
        nested, _ = summarise_bags(TestCursor([(cell,)]), query, "test")
        flat, _ = summarise_bags(TestCursor([(explicit,)]), query, "test")
        if nested != flat:
            raise RuntimeError(f"Q{query} expanded bag/PSD self-test failed")
    print("[SELF-TEST] bag multiplicities, factor PSD, Q92/Q94 expansion: PASS", flush=True)



def configure_cli():
    import argparse
    import re
    global SERVER_ROOT, INPUT_ROOT, DB_PATH, DB_SCHEMA, QUERY_IDS, DATASETS
    global MAX_WORKERS, DUCKDB_THREADS, DUCKDB_MEMORY_PER_WORKER
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--db', type=Path, required=True, help='Annotated DuckDB database')
    parser.add_argument('--queries-root', type=Path, required=True, help='Directory containing Q3, Q5, etc.')
    parser.add_argument('--queries', nargs='+', type=int, required=True, help='Explicit query IDs, e.g. 3 5 28')
    parser.add_argument('--pools', nargs='+', choices=('normal', 's1', 's2'), default=['normal', 's1', 's2'])
    parser.add_argument('--schema', default='tpcds')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2, help='DuckDB threads per worker')
    parser.add_argument('--memory-per-worker', default='8GB', help='DuckDB limit, not a total Python process limit')
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error('--db must be an existing annotated DuckDB file')
    if not args.queries_root.is_dir():
        parser.error('--queries-root must be an existing directory')
    if min(args.workers, args.threads, *args.queries) < 1:
        parser.error('Queries, workers and threads must be positive')
    if len(set(args.queries)) != len(args.queries) or len(set(args.pools)) != len(args.pools):
        parser.error('Duplicate queries or pools')
    if not args.schema.strip():
        parser.error('Schema cannot be empty')
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?\s*(?:KB|MB|GB|TB)', args.memory_per_worker, re.I) or float(re.match(r'[0-9.]+', args.memory_per_worker).group()) <= 0:
        parser.error('Memory must be positive, e.g. 8GB')
    INPUT_ROOT = args.queries_root.resolve()
    SERVER_ROOT = INPUT_ROOT.parent
    DB_PATH = args.db.resolve()
    DB_SCHEMA = args.schema
    QUERY_IDS = tuple(args.queries)
    DATASETS = tuple(args.pools)
    MAX_WORKERS = args.workers
    DUCKDB_THREADS = args.threads
    DUCKDB_MEMORY_PER_WORKER = args.memory_per_worker
    for query in QUERY_IDS:
        for pool in DATASETS:
            folder = INPUT_ROOT / f'Q{query}' / f'q{query}_{POOL_SUFFIX[pool]}'
            if not folder.is_dir():
                parser.error(f'Missing pool directory: {folder}')

def main() -> int:
    configure_cli()
    run_bag_self_test()
    if not DB_PATH.is_file() or not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Check DB_PATH={DB_PATH} and INPUT_ROOT={INPUT_ROOT}")
    if not QUERY_IDS or len(set(QUERY_IDS)) != len(QUERY_IDS):
        raise ValueError("Invalid query selection")
    if not DATASETS or set(DATASETS) - POOL_SUFFIX.keys() or MAX_WORKERS < 1:
        raise ValueError("Invalid dataset/worker selection")
    # Same lock as the compact generator: never read main files while it writes.
    with (INPUT_ROOT / ".materialiser.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = database_fingerprint()
        code_hash = computation_fingerprint()
        print(f"ROOT={INPUT_ROOT}\nQUERIES={QUERY_IDS}\nPOOLS={DATASETS}", flush=True)
        print("Bags: weighted CountSketch 512/1024. PSD: full exact counts; no SVD.", flush=True)
        print("Q28 Atomic bag: current r14 selected population when needed; scope is stored.", flush=True)
        print("Original main Parquets, SQL, metadata and splits are never replaced.", flush=True)
        with contextlib.ExitStack() as stack:
            scratch = stack.enter_context(tempfile.TemporaryDirectory(prefix=".bag_psd_tmp_", dir=SERVER_ROOT))
            connection = stack.enter_context(duckdb.connect(str(DB_PATH), read_only=True, config={
                "threads": DUCKDB_THREADS, "memory_limit": DUCKDB_MEMORY_PER_WORKER,
                "temp_directory": str(Path(scratch) / "parent")}))
            connection.execute("SET schema = '" + DB_SCHEMA.replace("'", "''") + "'")
            executor = None
            if MAX_WORKERS > 1:
                executor = stack.enter_context(concurrent.futures.ProcessPoolExecutor(
                    max_workers=MAX_WORKERS, mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialise_bag_worker, initargs=(str(DB_PATH), scratch, DB_SCHEMA, DUCKDB_THREADS, DUCKDB_MEMORY_PER_WORKER)))
            for query in QUERY_IDS:
                for pool in DATASETS:
                    process_bag_pool(connection, executor, query, pool, stamp, code_hash)
    print("[DONE] Selected bag/PSD sidecars completed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[STOPPED] Completed sidecars retained; do not evaluate a pool without its success marker.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)

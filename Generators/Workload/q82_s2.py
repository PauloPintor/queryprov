#!/usr/bin/env python3
from __future__ import annotations

import argparse

"""
q82_s2_generator.py

Generates the final S2 structural anomalies for TPC-DS Q82.

Final S2 subtypes
-----------------
- q82_s2_join_add_customer
- q82_s2_join_add_store
- q82_s2_where_remove_quantity

Methodological choices
----------------------
- The same deterministic sample of already-generated normal Q82 executions is
  used for every S2 subtype.
- Parameters are never filtered according to the magnitude of the mutation's
  effect. This avoids post-hoc instance selection.
- The generator mutates the existing normal SQL files directly. Consequently,
  price, date, manufacturer and all remaining query components are preserved
  exactly as generated in the normal workload.
- Every generated query must be executable and non-empty. A failure aborts the
  run instead of silently replacing the source instance with a more favourable
  one.
- Provenance shape is checked explicitly before bulk generation:
    normal / where_remove_quantity -> 4 tokens per witness
    join_add_customer / join_add_store -> 5 tokens per witness
"""

import csv
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# Linux configuration used for diagnostics.
DB_PATH = None  # Set by command-line arguments.
DB_SCHEMA = "tpcds"

Q82_NORMAL_DIR = None  # Set by command-line arguments.

# Write to a writable location. Change this to the external-disk path used
# during generation, for example:
# or:
#   Path("/run/media/paulo/Provenance/TPC-DS/Queries/Q82/q82_s2_anomalies")
OUT_S2 = None  # Set by command-line arguments.

SEED = 42
N_S2_PER_TYPE = 500

# Full provenance is expensive. It is needed only for a small, deterministic
# pre-generation shape check.
N_PROV_SHAPE_CHECKS_PER_TYPE = 3
MAX_WITNESSES_CHECKED_PER_QUERY = 100

S2_SUBTYPES = [
    "q82_s2_join_add_customer",
    "q82_s2_join_add_store",
    "q82_s2_where_remove_quantity",
]

S2_INFO: Dict[str, dict] = {
    "q82_s2_join_add_customer": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "intent_preservation": "high",
        "expected_witness_size": 5,
        "description": (
            "Adds customer through store_sales.ss_customer_sk = "
            "customer.c_customer_sk and includes customer provenance in each "
            "witness. Price, date, manufacturer and inventory predicates remain "
            "unchanged."
        ),
    },
    "q82_s2_join_add_store": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "intent_preservation": "high",
        "expected_witness_size": 5,
        "description": (
            "Adds store through store_sales.ss_store_sk = store.s_store_sk and "
            "includes store provenance in each witness. Price, date, "
            "manufacturer and inventory predicates remain unchanged."
        ),
    },
    "q82_s2_where_remove_quantity": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "intent_preservation": "medium",
        "expected_witness_size": 4,
        "description": (
            "Removes only the contextual inv_quantity_on_hand range predicate. "
            "The S1 dimensions price, date and manufacturer remain unchanged, "
            "and inventory remains joined and represented in provenance."
        ),
    },
}


# ============================================================
# NORMAL SOURCE LOADING
# ============================================================

def load_normal_metadata(normal_dir: Path) -> Optional[pd.DataFrame]:
    metadata_path = normal_dir / "metadata.csv"

    if not metadata_path.exists():
        return None

    df = pd.read_csv(metadata_path)

    if df.empty:
        return None

    file_col = None

    for candidate in ("file", "filename", "query_file", "sql_file"):
        if candidate in df.columns:
            file_col = candidate
            break

    if file_col is None:
        print(
            "[WARN] Normal metadata exists but has no recognised SQL filename "
            "column. Falling back to directory listing."
        )
        return None

    df = df.copy()
    df["_source_file"] = df[file_col].astype(str)
    df["_source_path"] = df["_source_file"].map(lambda x: normal_dir / x)

    missing = df[~df["_source_path"].map(Path.exists)]

    if not missing.empty:
        sample = missing["_source_path"].astype(str).head(5).tolist()
        raise FileNotFoundError(
            "Some SQL files referenced by normal metadata do not exist. "
            f"Examples: {sample}"
        )

    # A source execution appears only once in the candidate population.
    df = df.drop_duplicates(subset=["_source_file"]).reset_index(drop=True)

    return df


def list_normal_sources(normal_dir: Path) -> pd.DataFrame:
    files = sorted(
        p
        for p in normal_dir.glob("*.sql")
        if p.is_file() and not p.name.startswith("._")
    )

    if not files:
        raise FileNotFoundError(
            f"No normal Q82 SQL files found in {normal_dir}"
        )

    return pd.DataFrame(
        {
            "_source_file": [p.name for p in files],
            "_source_path": files,
        }
    )


def select_common_normal_sources(
    normal_dir: Path,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select one deterministic, unconditioned normal sample.

    The same source rows are reused for every S2 subtype. No cardinality,
    provenance size or mutation-effect statistic is used for selection.
    """

    df = load_normal_metadata(normal_dir)

    if df is None:
        print("[SOURCES] using normal SQL directory listing")
        df = list_normal_sources(normal_dir)
    else:
        print(
            f"[SOURCES] using normal metadata: "
            f"{normal_dir / 'metadata.csv'}"
        )

    if len(df) < n:
        raise ValueError(
            f"Need {n} distinct normal source queries, but only {len(df)} "
            "are available."
        )

    rng = np.random.default_rng(seed)
    positions = rng.choice(len(df), size=n, replace=False)

    selected = df.iloc[positions].copy().reset_index(drop=True)
    selected["_selection_index"] = np.arange(len(selected), dtype=np.int64)

    print(
        f"[SOURCES] selected={len(selected):,} "
        f"from available={len(df):,} seed={seed}"
    )

    return selected


# ============================================================
# SQL MUTATION HELPERS
# ============================================================

PROV_EXPR_RE = re.compile(
    r"""
    array_agg
    \s*\(
        \s*list_value
        \s*\(
            (?P<args>.*?)
        \)
    \s*\)
    \s*(?:as\s+)?prov
    """,
    flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

QUANTITY_PREDICATE_RE = re.compile(
    r"""
    \s+
    and
    \s+
    (?:inventory\.)?inv_quantity_on_hand
    \s+between\s+
    [-+]?\d+(?:\.\d+)?
    \s+and\s+
    [-+]?\d+(?:\.\d+)?
    """,
    flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def normalise_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split()).lower()


def add_relation_to_from(
    sql: str,
    relation_sql: str,
) -> str:
    """
    Append a relation to the first top-level FROM list before WHERE.

    Q82 has a single main FROM clause containing item, inventory, date_dim and
    store_sales. The mutation is deliberately rejected if that expected shape
    is not found.
    """

    match = re.search(
        r"\bfrom\b(?P<body>.*?)\bwhere\b",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match is None:
        raise ValueError("Could not locate Q82 FROM ... WHERE block.")

    body = match.group("body")

    if re.search(r"\bstore_sales\b", body, flags=re.IGNORECASE) is None:
        raise ValueError(
            "Expected store_sales in the Q82 FROM block, but it was not found."
        )

    new_body = body.rstrip() + f",\n     {relation_sql}\n"

    return sql[: match.start("body")] + new_body + sql[match.end("body") :]


def add_predicate_before_group_by(
    sql: str,
    predicate_sql: str,
) -> str:
    match = re.search(
        r"\bgroup\s+by\b",
        sql,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError("Could not locate Q82 GROUP BY clause.")

    insertion = f"\n  and {predicate_sql}\n"

    return sql[: match.start()] + insertion + sql[match.start() :]


def add_provenance_token(
    sql: str,
    token_sql: str,
) -> str:
    matches = list(PROV_EXPR_RE.finditer(sql))

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one array_agg(list_value(...)) provenance "
            f"expression, found {len(matches)}."
        )

    def replacement(match: re.Match[str]) -> str:
        args = match.group("args").strip()
        return (
            "array_agg(\n"
            "           list_value(\n"
            f"               {args},\n"
            f"               {token_sql}\n"
            "           )\n"
            "       ) as prov"
        )

    mutated, count = PROV_EXPR_RE.subn(
        replacement,
        sql,
        count=1,
    )

    if count != 1:
        raise RuntimeError("Failed to add provenance token.")

    return mutated


def remove_quantity_predicate(sql: str) -> str:
    mutated, count = QUANTITY_PREDICATE_RE.subn(
        "",
        sql,
        count=1,
    )

    if count != 1:
        raise ValueError(
            "Expected exactly one inv_quantity_on_hand BETWEEN predicate."
        )

    return mutated


def mutate_join_add_customer(normal_sql: str) -> str:
    sql = add_relation_to_from(normal_sql, "customer c")
    sql = add_predicate_before_group_by(
        sql,
        "store_sales.ss_customer_sk = c.c_customer_sk",
    )
    sql = add_provenance_token(sql, "c.prov")
    return sql


def mutate_join_add_store(normal_sql: str) -> str:
    sql = add_relation_to_from(normal_sql, "store s")
    sql = add_predicate_before_group_by(
        sql,
        "store_sales.ss_store_sk = s.s_store_sk",
    )
    sql = add_provenance_token(sql, "s.prov")
    return sql


def mutate_where_remove_quantity(normal_sql: str) -> str:
    return remove_quantity_predicate(normal_sql)


MUTATORS: Dict[str, Callable[[str], str]] = {
    "q82_s2_join_add_customer": mutate_join_add_customer,
    "q82_s2_join_add_store": mutate_join_add_store,
    "q82_s2_where_remove_quantity": mutate_where_remove_quantity,
}


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def require_pattern(sql: str, pattern: str, description: str) -> None:
    if re.search(pattern, sql, flags=re.IGNORECASE | re.DOTALL) is None:
        raise ValueError(f"Missing required SQL component: {description}")


def validate_preserved_s1_structure(sql: str) -> None:
    """
    Q82 S1 is defined over price, date and manufacturer.

    These checks ensure that no S2 mutator accidentally removes those logical
    units. They are intentionally semantic-pattern checks rather than checks for
    specific sampled values.
    """

    require_pattern(
        sql,
        r"\bi_current_price\b\s+between\b",
        "price range predicate",
    )
    require_pattern(
        sql,
        r"\bd_date\b\s+between\b",
        "date range predicate",
    )
    require_pattern(
        sql,
        r"\bi_manufact_id\b\s+in\s*\(",
        "manufacturer IN predicate",
    )
    require_pattern(
        sql,
        r"\binv_item_sk\b\s*=\s*\bi_item_sk\b",
        "inventory-item join",
    )
    require_pattern(
        sql,
        r"\bd_date_sk\b\s*=\s*\binv_date_sk\b",
        "date-inventory join",
    )
    require_pattern(
        sql,
        r"\bss_item_sk\b\s*=\s*\bi_item_sk\b",
        "store-sales item-existence join",
    )


def validate_mutated_structure(
    subtype: str,
    normal_sql: str,
    mutated_sql: str,
) -> None:
    if normalise_sql(normal_sql) == normalise_sql(mutated_sql):
        raise ValueError(f"{subtype} is textually equivalent to the normal SQL.")

    validate_preserved_s1_structure(mutated_sql)

    if subtype == "q82_s2_join_add_customer":
        require_pattern(
            mutated_sql,
            r"\bcustomer\s+c\b",
            "customer relation",
        )
        require_pattern(
            mutated_sql,
            r"\bss_customer_sk\b\s*=\s*c\.c_customer_sk\b",
            "customer join predicate",
        )
        require_pattern(
            mutated_sql,
            r"\bc\.prov\b",
            "customer provenance token",
        )
        require_pattern(
            mutated_sql,
            r"\binv_quantity_on_hand\b\s+between\b",
            "preserved inventory quantity predicate",
        )

    elif subtype == "q82_s2_join_add_store":
        require_pattern(
            mutated_sql,
            r"\bstore\s+s\b",
            "store relation",
        )
        require_pattern(
            mutated_sql,
            r"\bss_store_sk\b\s*=\s*s\.s_store_sk\b",
            "store join predicate",
        )
        require_pattern(
            mutated_sql,
            r"\bs\.prov\b",
            "store provenance token",
        )
        require_pattern(
            mutated_sql,
            r"\binv_quantity_on_hand\b\s+between\b",
            "preserved inventory quantity predicate",
        )

    elif subtype == "q82_s2_where_remove_quantity":
        if re.search(
            r"\binv_quantity_on_hand\b\s+between\b",
            mutated_sql,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "where_remove_quantity still contains the removed predicate."
            )

        # Inventory remains part of both the relational expression and
        # provenance; only the range restriction is removed.
        require_pattern(
            mutated_sql,
            r"\binventory\b",
            "preserved inventory relation",
        )
        require_pattern(
            mutated_sql,
            r"\binventory\.prov\b|\binv\.prov\b",
            "preserved inventory provenance",
        )

    else:
        raise ValueError(f"Unknown subtype: {subtype}")


# ============================================================
# EXECUTION VALIDATION
# ============================================================

def replace_provenance_with_null(sql: str) -> str:
    """
    Replace the expensive provenance aggregate with a constant for count-only
    validation. Relational joins, filters, grouping, ordering and limits remain
    unchanged.
    """

    replaced, count = PROV_EXPR_RE.subn(
        "NULL AS prov",
        sql,
        count=1,
    )

    if count != 1:
        raise ValueError(
            "Could not replace the Q82 provenance expression for validation."
        )

    return replaced


def build_count_sql(sql: str) -> str:
    cheap_sql = replace_provenance_with_null(sql).strip().rstrip(";")

    return f"""
select count(*)
from (
{cheap_sql}
) q82_s2_count
"""


def validate_non_empty(
    con: duckdb.DuckDBPyConnection,
    sql: str,
) -> int:
    row = con.execute(build_count_sql(sql)).fetchone()

    if row is None:
        raise RuntimeError("Count validation returned no row.")

    nrows = int(row[0])

    if nrows <= 0:
        raise RuntimeError(
            "Generated S2 query is empty. The run is aborted rather than "
            "replacing this preselected normal source with another instance."
        )

    return nrows


# ============================================================
# PROVENANCE SHAPE VALIDATION
# ============================================================

def to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, tuple):
        return list(value)

    return value


def flatten_scalar_tokens(value: Any) -> list[Any]:
    value = to_python(value)

    if value is None:
        return []

    if isinstance(value, list):
        out: list[Any] = []

        for element in value:
            out.extend(flatten_scalar_tokens(element))

        return out

    return [value]


def extract_witness_sizes(
    prov_cell: Any,
    max_witnesses: int,
) -> list[int]:
    outer = to_python(prov_cell)

    if outer is None:
        return []

    if not isinstance(outer, list):
        outer = [outer]

    sizes: list[int] = []

    for witness in outer:
        witness = to_python(witness)

        if witness is None:
            continue

        if not isinstance(witness, list):
            witness = [witness]

        sizes.append(len(flatten_scalar_tokens(witness)))

        if len(sizes) >= max_witnesses:
            break

    return sizes


def validate_provenance_shape(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    expected_witness_size: int,
    context: str,
) -> None:
    df = con.execute(sql).fetchdf()

    if df.empty:
        raise RuntimeError(
            f"{context}: full provenance query returned an empty result."
        )

    prov_col = None

    for col in df.columns:
        if str(col).lower() == "prov":
            prov_col = col
            break

    if prov_col is None:
        raise RuntimeError(f"{context}: result has no prov column.")

    observed_sizes: list[int] = []

    for cell in df[prov_col].tolist():
        observed_sizes.extend(
            extract_witness_sizes(
                cell,
                max_witnesses=(
                    MAX_WITNESSES_CHECKED_PER_QUERY - len(observed_sizes)
                ),
            )
        )

        if len(observed_sizes) >= MAX_WITNESSES_CHECKED_PER_QUERY:
            break

    if not observed_sizes:
        raise RuntimeError(
            f"{context}: could not extract any provenance witness."
        )

    unexpected = sorted(
        {
            size
            for size in observed_sizes
            if size != expected_witness_size
        }
    )

    if unexpected:
        raise RuntimeError(
            f"{context}: expected witness size {expected_witness_size}, "
            f"observed sizes {sorted(set(observed_sizes))}."
        )

    print(
        f"[PROV OK] {context}: "
        f"checked={len(observed_sizes)} "
        f"witness_size={expected_witness_size}"
    )


def run_pre_generation_checks(
    con: duckdb.DuckDBPyConnection,
    sources: pd.DataFrame,
) -> None:
    n_checks = min(N_PROV_SHAPE_CHECKS_PER_TYPE, len(sources))

    print("\n[PRE-CHECK] normal provenance shape")

    for i in range(n_checks):
        row = sources.iloc[i]
        normal_sql = Path(row["_source_path"]).read_text(encoding="utf-8")

        validate_preserved_s1_structure(normal_sql)
        validate_provenance_shape(
            con,
            normal_sql,
            expected_witness_size=4,
            context=f"normal source={row['_source_file']}",
        )

    for subtype in S2_SUBTYPES:
        print(f"\n[PRE-CHECK] {subtype}")

        mutator = MUTATORS[subtype]
        expected_size = int(
            S2_INFO[subtype]["expected_witness_size"]
        )

        for i in range(n_checks):
            row = sources.iloc[i]
            normal_sql = Path(row["_source_path"]).read_text(
                encoding="utf-8"
            )
            mutated_sql = mutator(normal_sql)

            validate_mutated_structure(
                subtype,
                normal_sql,
                mutated_sql,
            )
            validate_non_empty(con, mutated_sql)
            validate_provenance_shape(
                con,
                mutated_sql,
                expected_witness_size=expected_size,
                context=(
                    f"{subtype} source={row['_source_file']}"
                ),
            )


# ============================================================
# METADATA
# ============================================================

def metadata_value(value: Any) -> Any:
    if pd.isna(value):
        return ""

    if isinstance(value, np.generic):
        return value.item()

    return value


def source_metadata_fields(row: pd.Series) -> dict:
    reserved = {
        "_source_file",
        "_source_path",
        "_selection_index",
    }

    out: dict = {}

    for key, value in row.items():
        if key in reserved:
            continue

        out[f"normal_{key}"] = metadata_value(value)

    return out


def write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No metadata rows were generated.")

    fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[WRITE] metadata={path} rows={len(rows):,}")


# ============================================================
# GENERATION
# ============================================================

def subtype_filename(subtype: str, index: int) -> str:
    short = subtype.replace("q82_s2_", "")
    return f"{short}_{index:05d}.sql"


def generate_subtype(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    sources: pd.DataFrame,
    all_metadata: list[dict],
) -> None:
    info = S2_INFO[subtype]
    mutator = MUTATORS[subtype]

    print(f"\n[GENERATE] {subtype} n={len(sources):,}")

    for i, source_row in sources.iterrows():
        source_path = Path(source_row["_source_path"])
        normal_sql = source_path.read_text(encoding="utf-8")
        mutated_sql = mutator(normal_sql)

        validate_mutated_structure(
            subtype,
            normal_sql,
            mutated_sql,
        )

        # No retries and no replacement source instances. This preserves the
        # common preselected sample across all subtypes.
        nrows = validate_non_empty(con, mutated_sql)

        filename = subtype_filename(subtype, i)
        output_path = OUT_S2 / filename

        output_path.write_text(
            mutated_sql.strip().rstrip(";") + ";\n",
            encoding="utf-8",
        )

        metadata_row = {
            "file": filename,
            "label": int(info["label"]),
            "anomaly_class": info["anomaly_class"],
            "scenario": subtype,
            "category": info["category"],
            "logical_unit": info["logical_unit"],
            "intent_preservation": info["intent_preservation"],
            "description": info["description"],
            "source_normal_file": str(source_row["_source_file"]),
            "source_selection_index": int(
                source_row["_selection_index"]
            ),
            "n_rows": int(nrows),
            **source_metadata_fields(source_row),
        }

        all_metadata.append(metadata_row)

        if (i + 1) % 100 == 0:
            print(
                f"  [GEN] {subtype}: "
                f"{i + 1}/{len(sources)}"
            )

    print(f"  [OK] {subtype}: {len(sources):,} queries")


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q82 S2 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q82.")
    p.add_argument("--schema", default=DB_SCHEMA)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-per-type", type=int, default=N_S2_PER_TYPE)
    p.add_argument("--normal-dir", type=Path, required=True, help="Original normal pool with its SQL and metadata, as required by this generator.")
    args = p.parse_args(argv)
    args.db = args.db.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.db.is_file():
        p.error(f"Database file does not exist: {args.db}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        p.error(f"Not a directory: {args.output_dir}")
    if args.seed < 0 or args.n_per_type <= 0 or getattr(args, "n_normal", 1) <= 0:
        p.error("Seed must be non-negative and query counts must be positive.")
    if not args.schema.strip():
        p.error("Schema must not be empty.")
    args.normal_dir = args.normal_dir.expanduser().resolve()
    if not args.normal_dir.is_dir():
        p.error(f"Normal pool does not exist: {args.normal_dir}")
    for name in ['q82_s2_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, N_S2_PER_TYPE, OUT_S2, Q82_NORMAL_DIR, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_S2_PER_TYPE = args.n_per_type
    OUT_S2 = args.output_dir / "q82_s2_anomalies"
    Q82_NORMAL_DIR = args.normal_dir


def main() -> None:
    configure_cli()
    print("=" * 92)
    print("[Q82 FINAL S2 GENERATOR]")
    print("=" * 92)
    print(f"DB_PATH={DB_PATH}")
    print(f"DB_SCHEMA={DB_SCHEMA}")
    print(f"Q82_NORMAL_DIR={Q82_NORMAL_DIR}")
    print(f"OUT_S2={OUT_S2}")
    print(f"N_S2_PER_TYPE={N_S2_PER_TYPE}")
    print(f"SEED={SEED}")
    print(f"S2_SUBTYPES={S2_SUBTYPES}")
    print("=" * 92)

    sources = select_common_normal_sources(
        normal_dir=Q82_NORMAL_DIR,
        n=N_S2_PER_TYPE,
        seed=SEED,
    )

    OUT_S2.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET search_path = ?", [DB_SCHEMA])

    try:
        run_pre_generation_checks(con, sources)

        all_metadata: list[dict] = []

        for subtype in S2_SUBTYPES:
            generate_subtype(
                con=con,
                subtype=subtype,
                sources=sources,
                all_metadata=all_metadata,
            )

        write_metadata(
            OUT_S2 / "metadata.csv",
            all_metadata,
        )

    finally:
        con.close()

    print("\n[DONE]")
    print(
        f"  queries   : "
        f"{N_S2_PER_TYPE * len(S2_SUBTYPES):,}"
    )
    print(f"  directory : {OUT_S2}")
    print(f"  metadata  : {OUT_S2 / 'metadata.csv'}")


if __name__ == "__main__":
    main()

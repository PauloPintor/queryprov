#!/usr/bin/env python3
from __future__ import annotations

import argparse

"""
q89_s2_generator.py

Final S2 structural-anomaly generator for TPC-DS Q89.

Main S2 subtypes
----------------
- q89_s2_join_add_customer
- q89_s2_join_add_promotion
- q89_s2_where_remove_year
- q89_s2_agg_remove_brand
- q89_s2_limit_1
- q89_s2_limit_none

Methodological guarantees
-------------------------
- The same deterministic sample of already-generated normal Q89 executions is
  used for every S2 subtype.
- Source executions are sampled without conditioning on result cardinality,
  provenance size, or mutation magnitude.
- No failed or weak source execution is silently replaced.
- The six normally sampled category-class pairs are preserved in every S2
  subtype.
- Full provenance shape is checked on a deterministic prefix before bulk
  generation.

Expected provenance witness sizes
---------------------------------
- normal / predicate / aggregation / limit mutations -> 4 tokens
- join_add_customer / join_add_promotion             -> 5 tokens
"""

import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DB_PATH = None  # Set by command-line arguments.
DB_SCHEMA = "tpcds"

SEED = 42
N_S2_PER_TYPE = 500
N_PAIRS = 6

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

Q89_NORMAL_DIR = None  # Set by command-line arguments.
OUT_S2 = None  # Set by command-line arguments.

N_PROV_SHAPE_CHECKS_PER_SUBTYPE = 2
MAX_WITNESSES_CHECKED_PER_QUERY = 500

S2_SUBTYPES = [
    "q89_s2_join_add_customer",
    "q89_s2_join_add_promotion",
    "q89_s2_where_remove_year",
    "q89_s2_agg_remove_brand",
    "q89_s2_limit_1",
    "q89_s2_limit_none",
]

S2_INFO: Dict[str, dict] = {
    "q89_s2_join_add_customer": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_witness_size": 5,
        "description": (
            "Adds customer through ss_customer_sk = c_customer_sk and "
            "appends customer provenance to each witness."
        ),
    },
    "q89_s2_join_add_promotion": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_witness_size": 5,
        "description": (
            "Adds promotion through ss_promo_sk = p_promo_sk and appends "
            "promotion provenance to each witness."
        ),
    },
    "q89_s2_where_remove_year": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "expected_witness_size": 4,
        "description": (
            "Removes the fixed d_year predicate while preserving all six "
            "normally sampled category-class pairs."
        ),
    },
    "q89_s2_agg_remove_brand": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_aggregation",
        "logical_unit": "aggregation-granularity change",
        "expected_witness_size": 4,
        "description": (
            "Removes i_brand from the GROUP BY and window partition, merging "
            "brand-level groups while preserving category, class, store, "
            "company, and month."
        ),
    },
    "q89_s2_limit_1": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_limit",
        "logical_unit": "result cardinality limit",
        "expected_witness_size": 4,
        "description": (
            "Replaces LIMIT 100 with LIMIT 1 without conditioning source "
            "selection on the normal result cardinality."
        ),
    },
    "q89_s2_limit_none": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_limit",
        "logical_unit": "result cardinality limit",
        "expected_witness_size": 4,
        "description": (
            "Removes LIMIT 100 and materialises every qualifying monthly "
            "group."
        ),
    },
}


# ============================================================
# PATHS AND SOURCE SAMPLE
# ============================================================

def assert_paths() -> None:
    db_path = Path(DB_PATH)

    if not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB database does not exist: {db_path}\n"
            "Set DUCKDB_PATH to the correct database path."
        )

    metadata_path = Q89_NORMAL_DIR / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Normal Q89 metadata does not exist: {metadata_path}"
        )

    OUT_S2.mkdir(parents=True, exist_ok=True)

    test_path = OUT_S2 / "_write_test.tmp"

    try:
        test_path.write_text("ok\n", encoding="utf-8")
        test_path.unlink()
    except Exception as error:
        raise RuntimeError(
            f"Cannot write to output directory: {OUT_S2}"
        ) from error


def load_normal_metadata() -> pd.DataFrame:
    metadata_path = Q89_NORMAL_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)

    required = ["file", "year"]

    for index in range(1, N_PAIRS + 1):
        required.extend([f"cat{index}", f"class{index}"])

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Normal Q89 metadata is missing columns: {missing}"
        )

    df = df.dropna(subset=required).copy()

    if df.empty:
        raise ValueError("Normal Q89 metadata contains no usable rows.")

    df["file"] = df["file"].astype(str)
    df["year"] = pd.to_numeric(
        df["year"],
        errors="raise",
    ).astype(int)

    for index in range(1, N_PAIRS + 1):
        df[f"cat{index}"] = df[f"cat{index}"].astype(str)
        df[f"class{index}"] = df[f"class{index}"].astype(str)

    if "n_rows" in df.columns:
        df["n_rows"] = pd.to_numeric(
            df["n_rows"],
            errors="coerce",
        )

    df = df.drop_duplicates(subset=["file"]).reset_index(drop=True)

    return df


def select_common_normal_sources() -> pd.DataFrame:
    """
    Select one deterministic, unconditioned sample shared by all subtypes.
    """

    df = load_normal_metadata()

    if len(df) < N_S2_PER_TYPE:
        raise ValueError(
            f"Need {N_S2_PER_TYPE} distinct normal executions, "
            f"but metadata contains only {len(df)}."
        )

    rng = np.random.default_rng(SEED)
    positions = rng.choice(
        len(df),
        size=N_S2_PER_TYPE,
        replace=False,
    )

    selected = df.iloc[positions].copy().reset_index(drop=True)
    selected["_selection_index"] = np.arange(
        len(selected),
        dtype=np.int64,
    )

    print(
        f"[SOURCES] selected={len(selected):,} "
        f"from available={len(df):,} seed={SEED}"
    )

    return selected


# ============================================================
# SQL
# ============================================================

def sql_escape(value: str) -> str:
    return str(value).replace("'", "''")


def prepare_params(row: pd.Series | dict) -> dict:
    params = {
        "year": int(row["year"]),
    }

    for index in range(1, N_PAIRS + 1):
        params[f"cat{index}"] = sql_escape(
            str(row[f"cat{index}"])
        )
        params[f"class{index}"] = sql_escape(
            str(row[f"class{index}"])
        )

    return params


def pair_predicate_sql() -> str:
    branches: list[str] = []

    for index in range(1, N_PAIRS + 1):
        branches.append(
            "(\n"
            f"          i.i_category = '{{cat{index}}}'\n"
            f"      and i.i_class = '{{class{index}}}'\n"
            "      )"
        )

    return "(\n      " + "\n   or ".join(branches) + "\n  )"


def subtype_spec(subtype: str) -> dict:
    add_customer = subtype == "q89_s2_join_add_customer"
    add_promotion = subtype == "q89_s2_join_add_promotion"
    remove_year = subtype == "q89_s2_where_remove_year"
    remove_brand = subtype == "q89_s2_agg_remove_brand"

    relations = [
        "item i",
        "store_sales ss",
        "date_dim d",
        "store s",
    ]

    provenance = [
        "i.prov",
        "ss.prov",
        "d.prov",
        "s.prov",
    ]

    predicates = [
        "ss.ss_item_sk = i.i_item_sk",
        "ss.ss_sold_date_sk = d.d_date_sk",
        "ss.ss_store_sk = s.s_store_sk",
    ]

    if not remove_year:
        predicates.append("d.d_year = {year}")

    if add_customer:
        relations.append("customer c")
        provenance.append("c.prov")
        predicates.append(
            "ss.ss_customer_sk = c.c_customer_sk"
        )

    if add_promotion:
        relations.append("promotion p")
        provenance.append("p.prov")
        predicates.append(
            "ss.ss_promo_sk = p.p_promo_sk"
        )

    predicates.append(pair_predicate_sql())

    if subtype == "q89_s2_limit_1":
        limit: Optional[int] = 1
    elif subtype == "q89_s2_limit_none":
        limit = None
    else:
        limit = 100

    return {
        "relations": relations,
        "provenance": provenance,
        "predicates": predicates,
        "remove_brand": remove_brand,
        "limit": limit,
    }


def build_query_sql(
    subtype: str,
    with_provenance: bool,
    apply_limit: bool,
) -> str:
    spec = subtype_spec(subtype)

    relations_sql = ",\n         ".join(spec["relations"])
    predicates_sql = "\n      and ".join(spec["predicates"])

    if spec["remove_brand"]:
        brand_select = "cast(null as varchar) as i_brand"
        group_by = [
            "i.i_category",
            "i.i_class",
            "s.s_store_name",
            "s.s_company_name",
            "d.d_moy",
        ]
        partition_by = [
            "i.i_category",
            "s.s_store_name",
            "s.s_company_name",
        ]
    else:
        brand_select = "i.i_brand as i_brand"
        group_by = [
            "i.i_category",
            "i.i_class",
            "i.i_brand",
            "s.s_store_name",
            "s.s_company_name",
            "d.d_moy",
        ]
        partition_by = [
            "i.i_category",
            "i.i_brand",
            "s.s_store_name",
            "s.s_company_name",
        ]

    select_expressions = [
        "i.i_category as i_category",
        "i.i_class as i_class",
        brand_select,
        "s.s_store_name as s_store_name",
        "s.s_company_name as s_company_name",
        "d.d_moy as d_moy",
        "sum(ss.ss_sales_price) as sum_sales",
        (
            "avg(sum(ss.ss_sales_price)) over "
            f"(partition by {', '.join(partition_by)}) "
            "as avg_monthly_sales"
        ),
    ]

    if with_provenance:
        witness_args = ",\n                    ".join(
            spec["provenance"]
        )

        select_expressions.append(
            "array_agg(\n"
            "                list_value(\n"
            f"                    {witness_args}\n"
            "                )\n"
            "            ) as prov"
        )

    select_sql = ",\n            ".join(select_expressions)
    group_by_sql = ",\n        ".join(group_by)

    limit_sql = ""

    if apply_limit and spec["limit"] is not None:
        limit_sql = f"\nlimit {int(spec['limit'])}"

    return f"""
select *
from (
    select
            {select_sql}
    from {relations_sql}
    where {predicates_sql}
    group by
        {group_by_sql}
) tmp1
where case
        when avg_monthly_sales <> 0
        then abs(sum_sales - avg_monthly_sales) / avg_monthly_sales
        else null
      end > 0.1
order by
    sum_sales - avg_monthly_sales,
    s_store_name
{limit_sql}
"""


def render_sql(
    subtype: str,
    row: pd.Series | dict,
    with_provenance: bool = True,
    apply_limit: bool = True,
) -> str:
    template = build_query_sql(
        subtype=subtype,
        with_provenance=with_provenance,
        apply_limit=apply_limit,
    )

    return template.format(**prepare_params(row))


# ============================================================
# STATIC VALIDATION
# ============================================================

def normalised_sql(sql: str) -> str:
    return " ".join(
        sql.strip().rstrip(";").lower().split()
    )


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise ValueError(message)


def validate_structure(
    subtype: str,
    sql: str,
) -> None:
    lower = normalised_sql(sql)

    require(
        lower.count("i.i_category =") == N_PAIRS,
        f"{subtype}: expected {N_PAIRS} category predicates, "
        f"found {lower.count('i.i_category =')}.",
    )
    require(
        lower.count("i.i_class =") == N_PAIRS,
        f"{subtype}: expected {N_PAIRS} class predicates, "
        f"found {lower.count('i.i_class =')}.",
    )

    require(
        "ss.ss_item_sk = i.i_item_sk" in lower,
        f"{subtype}: item join is missing.",
    )
    require(
        "ss.ss_sold_date_sk = d.d_date_sk" in lower,
        f"{subtype}: date join is missing.",
    )
    require(
        "ss.ss_store_sk = s.s_store_sk" in lower,
        f"{subtype}: store join is missing.",
    )

    if subtype == "q89_s2_join_add_customer":
        require(
            "customer c" in lower,
            "join_add_customer: customer relation is missing.",
        )
        require(
            "ss.ss_customer_sk = c.c_customer_sk" in lower,
            "join_add_customer: customer join is missing.",
        )
        require(
            re.search(
                r"(?<![a-z0-9_])c\.prov(?![a-z0-9_])",
                lower,
            )
            is not None,
            "join_add_customer: customer provenance is missing.",
        )

    elif subtype == "q89_s2_join_add_promotion":
        require(
            "promotion p" in lower,
            "join_add_promotion: promotion relation is missing.",
        )
        require(
            "ss.ss_promo_sk = p.p_promo_sk" in lower,
            "join_add_promotion: promotion join is missing.",
        )
        require(
            re.search(
                r"(?<![a-z0-9_])p\.prov(?![a-z0-9_])",
                lower,
            )
            is not None,
            "join_add_promotion: promotion provenance is missing.",
        )

    elif subtype == "q89_s2_where_remove_year":
        require(
            "d.d_year =" not in lower,
            "where_remove_year: year predicate remains.",
        )

    elif subtype == "q89_s2_agg_remove_brand":
        require(
            "cast(null as varchar) as i_brand" in lower,
            "agg_remove_brand: projected brand placeholder is missing.",
        )
        require(
            (
                "group by i.i_category, i.i_class, "
                "s.s_store_name, s.s_company_name, d.d_moy"
            )
            in lower,
            "agg_remove_brand: coarser GROUP BY was not generated.",
        )
        require(
            (
                "partition by i.i_category, "
                "s.s_store_name, s.s_company_name"
            )
            in lower,
            "agg_remove_brand: brand remains in the window partition.",
        )

    elif subtype == "q89_s2_limit_1":
        require(
            lower.endswith("limit 1"),
            "limit_1: query does not end with LIMIT 1.",
        )

    elif subtype == "q89_s2_limit_none":
        require(
            " limit " not in lower,
            "limit_none: a LIMIT clause remains.",
        )

    if subtype != "q89_s2_where_remove_year":
        require(
            "d.d_year =" in lower,
            f"{subtype}: year predicate is missing.",
        )


# ============================================================
# EXECUTION AND PROVENANCE VALIDATION
# ============================================================

def count_rows(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    row: pd.Series,
    apply_limit: bool,
) -> int:
    sql = render_sql(
        subtype=subtype,
        row=row,
        with_provenance=False,
        apply_limit=apply_limit,
    ).strip().rstrip(";")

    count_sql = f"""
select count(*)
from (
{sql}
) q89_s2_count
"""

    result = con.execute(count_sql).fetchone()

    if result is None:
        raise RuntimeError(
            f"{subtype}: count query returned no row."
        )

    return int(result[0])


def as_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)

    return value


def flatten_scalar_tokens(value: Any) -> list[Any]:
    value = as_python(value)

    if value is None:
        return []

    if isinstance(value, list):
        output: list[Any] = []

        for element in value:
            output.extend(flatten_scalar_tokens(element))

        return output

    return [value]


def extract_witnesses(prov_cell: Any) -> list[list[Any]]:
    outer = as_python(prov_cell)

    if outer is None:
        return []

    if not isinstance(outer, list):
        outer = [outer]

    witnesses: list[list[Any]] = []

    for witness in outer:
        witness = as_python(witness)

        if witness is None:
            continue

        witnesses.append(flatten_scalar_tokens(witness))

    return witnesses


def provenance_witness_sizes(
    df: pd.DataFrame,
) -> list[int]:
    if "prov" not in df.columns:
        raise RuntimeError("Result has no prov column.")

    sizes: list[int] = []

    for prov_cell in df["prov"].tolist():
        for witness in extract_witnesses(prov_cell):
            sizes.append(len(witness))

            if len(sizes) >= MAX_WITNESSES_CHECKED_PER_QUERY:
                return sizes

    return sizes


def validate_full_provenance(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    row: pd.Series,
) -> None:
    sql = render_sql(
        subtype=subtype,
        row=row,
        with_provenance=True,
        apply_limit=True,
    )

    df = con.execute(sql).fetchdf()

    if df.empty:
        raise RuntimeError(
            f"{subtype}: full provenance query returned no rows."
        )

    sizes = provenance_witness_sizes(df)

    if not sizes:
        raise RuntimeError(
            f"{subtype}: no provenance witnesses were extracted."
        )

    expected = int(
        S2_INFO[subtype]["expected_witness_size"]
    )
    observed = sorted(set(sizes))

    if observed != [expected]:
        raise RuntimeError(
            f"{subtype}: expected witness size {expected}, "
            f"observed {observed}."
        )

    print(
        f"[PROV OK] {subtype} "
        f"source={row['file']} "
        f"checked={len(sizes)} "
        f"witness_size={expected}"
    )


def run_pre_generation_checks(
    con: duckdb.DuckDBPyConnection,
    sources: pd.DataFrame,
) -> None:
    n_checks = min(
        N_PROV_SHAPE_CHECKS_PER_SUBTYPE,
        len(sources),
    )

    for subtype in S2_SUBTYPES:
        print(f"\n[PRE-CHECK] {subtype}")

        for source_index in range(n_checks):
            row = sources.iloc[source_index]
            sql = render_sql(
                subtype=subtype,
                row=row,
                with_provenance=True,
                apply_limit=True,
            )

            validate_structure(subtype, sql)

            limited_nrows = count_rows(
                con,
                subtype,
                row,
                apply_limit=True,
            )

            if limited_nrows <= 0:
                raise RuntimeError(
                    f"{subtype}: source {row['file']} produced "
                    "an empty result during pre-check."
                )

            validate_full_provenance(
                con,
                subtype,
                row,
            )


# ============================================================
# OUTPUT
# ============================================================

def subtype_filename(
    subtype: str,
    index: int,
) -> str:
    short_name = subtype.replace("q89_s2_", "")

    return f"s2_{short_name}_{index:05d}.sql"


def metadata_parameter_values(
    row: pd.Series,
) -> dict:
    output = {
        "year": int(row["year"]),
    }

    for index in range(1, N_PAIRS + 1):
        output[f"cat{index}"] = str(row[f"cat{index}"])
        output[f"class{index}"] = str(row[f"class{index}"])

    if "n_rows" in row.index and not pd.isna(row["n_rows"]):
        output["normal_n_rows_unbounded"] = int(row["n_rows"])

    return output


def write_metadata(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise ValueError("No S2 metadata rows were generated.")

    fieldnames = sorted(
        {
            field
            for row in rows
            for field in row
        }
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[WRITE] {path} rows={len(rows):,}")


def generate_subtype(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    sources: pd.DataFrame,
    metadata_rows: list[dict],
) -> None:
    info = S2_INFO[subtype]

    print(f"\n[GENERATE] {subtype} n={len(sources):,}")

    for index, row in sources.iterrows():
        sql = render_sql(
            subtype=subtype,
            row=row,
            with_provenance=True,
            apply_limit=True,
        )

        validate_structure(subtype, sql)

        # These checks never trigger resampling. Any failure aborts the run.
        limited_nrows = count_rows(
            con,
            subtype,
            row,
            apply_limit=True,
        )
        unbounded_nrows = count_rows(
            con,
            subtype,
            row,
            apply_limit=False,
        )

        if limited_nrows <= 0:
            raise RuntimeError(
                f"{subtype}: preselected source {row['file']} produced "
                "an empty result. The run is aborted and the source is "
                "not replaced."
            )

        filename = subtype_filename(
            subtype,
            index,
        )

        (OUT_S2 / filename).write_text(
            sql.strip().rstrip(";") + ";\n",
            encoding="utf-8",
        )

        metadata_rows.append(
            {
                "file": filename,
                "label": int(info["label"]),
                "anomaly_class": info["anomaly_class"],
                "scenario": subtype,
                "category": info["category"],
                "logical_unit": info["logical_unit"],
                "description": info["description"],
                "source_normal_file": str(row["file"]),
                "source_selection_index": int(
                    row["_selection_index"]
                ),
                "n_rows": int(limited_nrows),
                "n_rows_unbounded": int(unbounded_nrows),
                "limit_active": int(
                    unbounded_nrows > limited_nrows
                ),
                **metadata_parameter_values(row),
            }
        )

        if (index + 1) % 100 == 0:
            print(
                f"  [GEN] {subtype}: "
                f"{index + 1}/{len(sources)}"
            )

    print(f"  [OK] {subtype}: {len(sources):,} queries")


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q89 S2 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q89.")
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
    for name in ['q89_s2_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, EXTERNAL_BASE_DIR, N_S2_PER_TYPE, OUT_S2, Q89_NORMAL_DIR, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_S2_PER_TYPE = args.n_per_type
    OUT_S2 = args.output_dir / "q89_s2_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir
    Q89_NORMAL_DIR = args.normal_dir


def main() -> None:
    configure_cli()
    print("=" * 92)
    print("[Q89 FINAL S2 GENERATOR]")
    print("=" * 92)
    print(f"DB_PATH={DB_PATH}")
    print(f"DB_SCHEMA={DB_SCHEMA}")
    print(f"Q89_NORMAL_DIR={Q89_NORMAL_DIR}")
    print(f"OUT_S2={OUT_S2}")
    print(f"SEED={SEED}")
    print(f"N_S2_PER_TYPE={N_S2_PER_TYPE}")
    print(f"S2_SUBTYPES={S2_SUBTYPES}")
    print("=" * 92)

    assert_paths()
    sources = select_common_normal_sources()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET search_path = ?", [DB_SCHEMA])

    try:
        run_pre_generation_checks(
            con,
            sources,
        )

        metadata_rows: list[dict] = []

        for subtype in S2_SUBTYPES:
            generate_subtype(
                con=con,
                subtype=subtype,
                sources=sources,
                metadata_rows=metadata_rows,
            )

        write_metadata(
            OUT_S2 / "metadata.csv",
            metadata_rows,
        )

    finally:
        con.close()

    total = N_S2_PER_TYPE * len(S2_SUBTYPES)

    print("\n[DONE]")
    print(f"  queries   : {total:,}")
    print(f"  directory : {OUT_S2}")
    print(f"  metadata  : {OUT_S2 / 'metadata.csv'}")


if __name__ == "__main__":
    main()

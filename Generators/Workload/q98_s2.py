#!/usr/bin/env python3
from __future__ import annotations

import argparse

import csv
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
N_CATEGORIES = 3

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

Q98_NORMAL_DIR = None  # Set by command-line arguments.
OUT_S2 = None  # Set by command-line arguments.

N_PROV_SHAPE_CHECKS_PER_SUBTYPE = 3
MAX_WITNESSES_CHECKED_PER_QUERY = 500

S2_SUBTYPES = [
    "q98_s2_join_add_customer",
    "q98_s2_join_add_store",
    "q98_s2_join_add_promotion",
    "q98_s2_limit_1",
    "q98_s2_limit_100",
]

S2_INFO: Dict[str, dict] = {
    "q98_s2_join_add_customer": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_witness_size": 4,
        "description": (
            "Adds customer through ss_customer_sk = c_customer_sk and appends "
            "customer provenance to each witness."
        ),
    },
    "q98_s2_join_add_store": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_witness_size": 4,
        "description": (
            "Adds store through ss_store_sk = s_store_sk and appends store "
            "provenance to each witness."
        ),
    },
    "q98_s2_join_add_promotion": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_witness_size": 4,
        "description": (
            "Adds promotion through ss_promo_sk = p_promo_sk and appends "
            "promotion provenance to each witness."
        ),
    },
    "q98_s2_limit_1": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_limit",
        "logical_unit": "result cardinality limit",
        "expected_witness_size": 3,
        "description": (
            "Adds LIMIT 1 to an originally unlimited query containing "
            "approximately 2500 ordered item groups."
        ),
    },
    "q98_s2_limit_100": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_limit",
        "logical_unit": "result cardinality limit",
        "expected_witness_size": 3,
        "description": (
            "Adds LIMIT 100 to an originally unlimited query containing "
            "approximately 2500 ordered item groups."
        ),
    },
}


# ============================================================
# PATHS AND NORMAL SOURCE SAMPLE
# ============================================================

def assert_paths() -> None:
    db_path = Path(DB_PATH)

    if not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB database does not exist: {db_path}\n"
            "Set DUCKDB_PATH to the correct database path."
        )

    metadata_path = Q98_NORMAL_DIR / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Normal Q98 metadata does not exist: {metadata_path}"
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
    metadata_path = Q98_NORMAL_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)

    required = [
        "file",
        "category1",
        "category2",
        "category3",
        "sdate",
        "window_days",
        "n_rows",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Normal Q98 metadata is missing columns: {missing}"
        )

    df = df.dropna(subset=required).copy()

    if df.empty:
        raise ValueError("Normal Q98 metadata contains no usable rows.")

    df["file"] = df["file"].astype(str)

    for index in range(1, N_CATEGORIES + 1):
        df[f"category{index}"] = df[
            f"category{index}"
        ].astype(str)

    df["sdate"] = pd.to_datetime(
        df["sdate"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    df["window_days"] = pd.to_numeric(
        df["window_days"],
        errors="raise",
    ).astype(int)
    df["n_rows"] = pd.to_numeric(
        df["n_rows"],
        errors="raise",
    ).astype(int)

    df = df.drop_duplicates(subset=["file"]).reset_index(drop=True)

    return df


def select_common_normal_sources() -> pd.DataFrame:
    """
    Select one deterministic, unconditioned normal sample shared by every
    structural subtype.
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
# SQL RENDERING
# ============================================================

def sql_escape(value: str) -> str:
    return str(value).replace("'", "''")


def prepare_params(row: pd.Series | dict) -> dict:
    return {
        "category1": sql_escape(str(row["category1"])),
        "category2": sql_escape(str(row["category2"])),
        "category3": sql_escape(str(row["category3"])),
        "sdate": str(row["sdate"]),
        "window_days": int(row["window_days"]),
    }


def subtype_spec(subtype: str) -> dict:
    add_customer = subtype == "q98_s2_join_add_customer"
    add_store = subtype == "q98_s2_join_add_store"
    add_promotion = subtype == "q98_s2_join_add_promotion"

    relations: list[tuple[str, Optional[str]]] = [
        ("store_sales ss", None),
        ("item i", "ss.ss_item_sk = i.i_item_sk"),
        ("date_dim d", "ss.ss_sold_date_sk = d.d_date_sk"),
    ]

    provenance = [
        "ss.prov",
        "i.prov",
        "d.prov",
    ]

    if add_customer:
        relations.append(
            (
                "customer c",
                "ss.ss_customer_sk = c.c_customer_sk",
            )
        )
        provenance.append("c.prov")

    if add_store:
        relations.append(
            (
                "store s",
                "ss.ss_store_sk = s.s_store_sk",
            )
        )
        provenance.append("s.prov")

    if add_promotion:
        relations.append(
            (
                "promotion p",
                "ss.ss_promo_sk = p.p_promo_sk",
            )
        )
        provenance.append("p.prov")

    if subtype == "q98_s2_limit_1":
        limit: Optional[int] = 1
    elif subtype == "q98_s2_limit_100":
        limit = 100
    else:
        limit = None

    return {
        "relations": relations,
        "provenance": provenance,
        "limit": limit,
    }


def build_from_sql(
    subtype: str,
) -> str:
    relations = subtype_spec(subtype)["relations"]

    lines = [relations[0][0]]

    for relation, predicate in relations[1:]:
        lines.extend(
            [
                f"join {relation}",
                f"  on {predicate}",
            ]
        )

    return "\n    ".join(lines)


def build_query_sql(
    subtype: str,
    with_provenance: bool,
    apply_limit: bool,
) -> str:
    spec = subtype_spec(subtype)
    from_sql = build_from_sql(subtype)

    select_expressions = [
        "i.i_item_id as i_item_id",
        "i.i_item_desc as i_item_desc",
        "i.i_category as i_category",
        "i.i_class as i_class",
        "i.i_current_price as i_current_price",
        "sum(ss.ss_ext_sales_price) as itemrevenue",
        (
            "sum(ss.ss_ext_sales_price) * 100.0 "
            "/ sum(sum(ss.ss_ext_sales_price)) over "
            "(partition by i.i_class) as revenueratio"
        ),
    ]

    if with_provenance:
        witness_args = ",\n                ".join(
            spec["provenance"]
        )

        select_expressions.append(
            "array_agg(\n"
            "        list_value(\n"
            f"                {witness_args}\n"
            "        )\n"
            "    ) as prov"
        )

    select_sql = ",\n    ".join(select_expressions)

    limit_sql = ""

    if apply_limit and spec["limit"] is not None:
        limit_sql = f"\nlimit {int(spec['limit'])}"

    return f"""
select
    {select_sql}
from {from_sql}
where i.i_category in (
        '{{category1}}',
        '{{category2}}',
        '{{category3}}'
      )
  and d.d_date between cast('{{sdate}}' as date)
                   and cast('{{sdate}}' as date)
                       + interval {{window_days}} day
group by
    i.i_item_id,
    i.i_item_desc,
    i.i_category,
    i.i_class,
    i.i_current_price
order by
    i_category,
    i_class,
    i_item_id,
    i_item_desc,
    revenueratio
{limit_sql}
"""


def build_contributor_count_sql(
    subtype: str,
) -> str:
    from_sql = build_from_sql(subtype)

    return f"""
select count(*) as n_contributors
from {from_sql}
where i.i_category in (
        '{{category1}}',
        '{{category2}}',
        '{{category3}}'
      )
  and d.d_date between cast('{{sdate}}' as date)
                   and cast('{{sdate}}' as date)
                       + interval {{window_days}} day
"""


def render_sql(
    sql: str,
    row: pd.Series | dict,
) -> str:
    return sql.format(**prepare_params(row))


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


def has_alias_prov(
    sql: str,
    alias: str,
) -> bool:
    return (
        re.search(
            rf"(?<![a-z0-9_]){re.escape(alias)}\.prov"
            rf"(?![a-z0-9_])",
            sql,
        )
        is not None
    )


def validate_structure(
    subtype: str,
    sql: str,
) -> None:
    lower = normalised_sql(sql)

    # All four S1 values remain represented.
    require(
        "i.i_category in (" in lower,
        f"{subtype}: category predicate is missing.",
    )
    require(
        lower.count("'") >= 8,
        f"{subtype}: rendered category/date literals appear incomplete.",
    )
    require(
        "d.d_date between" in lower,
        f"{subtype}: date-window predicate is missing.",
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
        "partition by i.i_class" in lower,
        f"{subtype}: normal class-level window partition is missing.",
    )
    require(
        (
            "group by i.i_item_id, i.i_item_desc, "
            "i.i_category, i.i_class, i.i_current_price"
        )
        in lower,
        f"{subtype}: normal item-level GROUP BY is missing.",
    )

    if subtype == "q98_s2_join_add_customer":
        require(
            "join customer c" in lower,
            "join_add_customer: customer relation is missing.",
        )
        require(
            "ss.ss_customer_sk = c.c_customer_sk" in lower,
            "join_add_customer: customer join is missing.",
        )
        require(
            has_alias_prov(lower, "c"),
            "join_add_customer: customer provenance is missing.",
        )

    elif subtype == "q98_s2_join_add_store":
        require(
            "join store s" in lower,
            "join_add_store: store relation is missing.",
        )
        require(
            "ss.ss_store_sk = s.s_store_sk" in lower,
            "join_add_store: store join is missing.",
        )
        require(
            has_alias_prov(lower, "s"),
            "join_add_store: store provenance is missing.",
        )

    elif subtype == "q98_s2_join_add_promotion":
        require(
            "join promotion p" in lower,
            "join_add_promotion: promotion relation is missing.",
        )
        require(
            "ss.ss_promo_sk = p.p_promo_sk" in lower,
            "join_add_promotion: promotion join is missing.",
        )
        require(
            has_alias_prov(lower, "p"),
            "join_add_promotion: promotion provenance is missing.",
        )

    elif subtype == "q98_s2_limit_1":
        require(
            lower.endswith("limit 1"),
            "limit_1: query does not end with LIMIT 1.",
        )

    elif subtype == "q98_s2_limit_100":
        require(
            lower.endswith("limit 100"),
            "limit_100: query does not end with LIMIT 100.",
        )


# ============================================================
# EXECUTION AND PROVENANCE VALIDATION
# ============================================================

def execute_scalar_count(
    con: duckdb.DuckDBPyConnection,
    sql: str,
) -> int:
    result = con.execute(sql).fetchone()

    if result is None:
        raise RuntimeError("Count query returned no row.")

    return int(result[0])


def contributor_count(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    row: pd.Series,
) -> int:
    return execute_scalar_count(
        con,
        render_sql(
            build_contributor_count_sql(subtype),
            row,
        ),
    )


def result_count(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    row: pd.Series,
    apply_limit: bool,
) -> int:
    sql = render_sql(
        build_query_sql(
            subtype=subtype,
            with_provenance=False,
            apply_limit=apply_limit,
        ),
        row,
    ).strip().rstrip(";")

    return execute_scalar_count(
        con,
        f"""
select count(*)
from (
{sql}
) q98_result_count
""",
    )


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

        witnesses.append(
            flatten_scalar_tokens(witness)
        )

    return witnesses


def validate_full_provenance(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    row: pd.Series,
) -> None:
    sql = render_sql(
        build_query_sql(
            subtype=subtype,
            with_provenance=True,
            apply_limit=True,
        ),
        row,
    )

    df = con.execute(sql).fetchdf()

    if df.empty:
        raise RuntimeError(
            f"{subtype}: full provenance query returned no rows."
        )

    expected = int(
        S2_INFO[subtype]["expected_witness_size"]
    )
    checked = 0

    for prov_cell in df["prov"].tolist():
        for witness in extract_witnesses(prov_cell):
            observed = len(witness)

            if observed != expected:
                raise RuntimeError(
                    f"{subtype}: expected witness size {expected}, "
                    f"observed {observed}."
                )

            checked += 1

            if checked >= MAX_WITNESSES_CHECKED_PER_QUERY:
                break

        if checked >= MAX_WITNESSES_CHECKED_PER_QUERY:
            break

    if checked == 0:
        raise RuntimeError(
            f"{subtype}: no provenance witnesses were extracted."
        )

    print(
        f"[PROV OK] {subtype} "
        f"source={row['file']} "
        f"checked={checked} "
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
                build_query_sql(
                    subtype=subtype,
                    with_provenance=True,
                    apply_limit=True,
                ),
                row,
            )

            validate_structure(subtype, sql)

            n_contributors = contributor_count(
                con,
                subtype,
                row,
            )
            n_unbounded = result_count(
                con,
                subtype,
                row,
                apply_limit=False,
            )
            n_materialised = result_count(
                con,
                subtype,
                row,
                apply_limit=True,
            )

            if n_contributors <= 0:
                raise RuntimeError(
                    f"{subtype}: source {row['file']} produced no "
                    "contributing tuples during pre-check."
                )

            if n_unbounded <= 0 or n_materialised <= 0:
                raise RuntimeError(
                    f"{subtype}: source {row['file']} produced an "
                    "empty result during pre-check."
                )

            validate_full_provenance(
                con,
                subtype,
                row,
            )


# ============================================================
# METADATA AND GENERATION
# ============================================================

def subtype_filename(
    subtype: str,
    index: int,
) -> str:
    short_name = subtype.replace("q98_s2_", "")

    return f"s2_{short_name}_{index:05d}.sql"


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
            build_query_sql(
                subtype=subtype,
                with_provenance=True,
                apply_limit=True,
            ),
            row,
        )

        validate_structure(subtype, sql)

        # No resampling occurs. Any invalid preselected source aborts the run.
        n_contributors = contributor_count(
            con,
            subtype,
            row,
        )
        n_unbounded = result_count(
            con,
            subtype,
            row,
            apply_limit=False,
        )
        n_materialised = result_count(
            con,
            subtype,
            row,
            apply_limit=True,
        )

        if (
            n_contributors <= 0
            or n_unbounded <= 0
            or n_materialised <= 0
        ):
            raise RuntimeError(
                f"{subtype}: preselected source {row['file']} produced "
                "an empty execution. The run is aborted and the source "
                "is not replaced."
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
                "category1": str(row["category1"]),
                "category2": str(row["category2"]),
                "category3": str(row["category3"]),
                "sdate": str(row["sdate"]),
                "window_days": int(row["window_days"]),
                # Matches the unbounded group-count semantics of the
                # original Q98 metadata.
                "n_rows": int(n_unbounded),
                "result_n_rows": int(n_materialised),
                "n_contributors": int(n_contributors),
                "normal_n_rows": int(row["n_rows"]),
                "limit_active": int(
                    n_unbounded > n_materialised
                ),
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
    p = argparse.ArgumentParser(description="Generate Q98 S2 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q98.")
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
    for name in ['q98_s2_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, EXTERNAL_BASE_DIR, N_S2_PER_TYPE, OUT_S2, Q98_NORMAL_DIR, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_S2_PER_TYPE = args.n_per_type
    EXTERNAL_BASE_DIR = args.output_dir
    OUT_S2 = args.output_dir / "q98_s2_anomalies"
    Q98_NORMAL_DIR = args.normal_dir


def main() -> None:
    configure_cli()
    print("=" * 96)
    print("[Q98 FINAL S2 GENERATOR]")
    print("=" * 96)
    print(f"DB_PATH={DB_PATH}")
    print(f"DB_SCHEMA={DB_SCHEMA}")
    print(f"Q98_NORMAL_DIR={Q98_NORMAL_DIR}")
    print(f"OUT_S2={OUT_S2}")
    print(f"SEED={SEED}")
    print(f"N_S2_PER_TYPE={N_S2_PER_TYPE}")
    print(f"S2_SUBTYPES={S2_SUBTYPES}")
    print("=" * 96)

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

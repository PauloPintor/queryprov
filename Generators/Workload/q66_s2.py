#!/usr/bin/env python3
from __future__ import annotations

import argparse

"""
Generate the final S2 structural-anomaly workload for TPC-DS Q66.

Retained S2 mutations
---------------------
- q66_s2_join_add_customer
- q66_s2_join_add_item
- q66_s2_where_remove_time
- q66_s2_agg_group_by_state
- q66_s2_limit_1

The sampled values of carrier, time, year, metric columns, and time window are
copied from the already-generated normal Q66 metadata. Consequently, the
anomaly is introduced only by the SQL mutation, not by an anomalous parameter
value.

Provenance semantics
--------------------
The normal Q66 representation builds one witness per contributing sales tuple:

    [sales, warehouse, date, time, ship_mode]

The two join-add mutations extend each witness with exactly the provenance token
of the newly joined dimension:

    join_add_customer -> [..., customer]
    join_add_item     -> [..., item]

Predicate, aggregation, and limit mutations do not invent or remove provenance
terms independently of SQL semantics. They retain the same per-tuple witnesses
and only change which tuples contribute, how witnesses are grouped, or which
output rows are materialised.
"""

import csv
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# Mac generation environment used by the original Q66 generator.
DB_PATH = None  # Set by command-line arguments.
DB_SCHEMA = "tpcds"

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.
Q66_NORMAL_DIR = None  # Set by command-line arguments.
OUT_S2 = None  # Set by command-line arguments.

# Linux alternatives:
# EXTERNAL_BASE_DIR = Path("/path/to/writable/TPC-DS/Queries/Q66")
# OUT_S2 = EXTERNAL_BASE_DIR / "q66_s2_anomalies"

SEED = 42
N_S2_PER_TYPE = 500

# Use the same sampled normal parameter combinations for every subtype. This
# isolates the structural mutation and creates balanced, matched S2 groups.
MATCH_PARAMETERS_ACROSS_SUBTYPES = True

# Execute the complete provenance query for the first N generated instances of
# each subtype and check the shape of every witness. This is deliberately small:
# count validation is applied to every query, while full provenance validation
# can be expensive.
VALIDATE_FULL_PROVENANCE = True
FULL_PROVENANCE_CHECKS_PER_SUBTYPE = 1

# False is safer. Existing files with the same names are overwritten. Set True
# only when the complete q66_s2_anomalies directory should be rebuilt.
CLEAN_OUTPUT_DIR = False

S2_SUBTYPES = [
    "q66_s2_join_add_customer",
    "q66_s2_join_add_item",
    "q66_s2_where_remove_time",
    "q66_s2_agg_group_by_state",
    "q66_s2_limit_1",
]

S2_INFO: Dict[str, dict] = {
    "q66_s2_join_add_customer": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "intent_preservation": "high",
        "expected_witness_size": 6,
        "description": (
            "Adds customer to both sales branches through the corresponding "
            "bill-customer foreign key and includes customer provenance in "
            "each witness."
        ),
    },
    "q66_s2_join_add_item": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "intent_preservation": "high",
        "expected_witness_size": 6,
        "description": (
            "Adds item to both sales branches through the corresponding "
            "item foreign key and includes item provenance in each witness."
        ),
    },
    "q66_s2_where_remove_time": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "intent_preservation": "medium",
        "expected_witness_size": 5,
        "description": (
            "Removes the contextual t_time range predicate from both sales "
            "branches while preserving the normally sampled carrier pair and "
            "all joins, including the time_dim join."
        ),
    },
    "q66_s2_agg_group_by_state": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_aggregation",
        "logical_unit": "aggregation granularity",
        "intent_preservation": "medium",
        "expected_witness_size": 5,
        "description": (
            "Changes the final aggregation granularity from warehouse to "
            "state/country, retaining carrier and year."
        ),
    },
    "q66_s2_limit_1": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_limit",
        "logical_unit": "result cardinality limit",
        "intent_preservation": "medium",
        "expected_witness_size": 5,
        "description": (
            "Changes the final LIMIT 100 to LIMIT 1 without changing joins, "
            "predicates, aggregation, or sampled parameters."
        ),
    },
}

MONTHS = [
    (1, "jan"),
    (2, "feb"),
    (3, "mar"),
    (4, "apr"),
    (5, "may"),
    (6, "jun"),
    (7, "jul"),
    (8, "aug"),
    (9, "sep"),
    (10, "oct"),
    (11, "nov"),
    (12, "dec"),
]

REQUIRED_METADATA_COLUMNS = [
    "smc1",
    "smc2",
    "year",
    "timeone",
    "time_window",
    "netone",
    "nettwo",
    "salesone",
    "salestwo",
]


# ============================================================
# METADATA / PARAMETER SAMPLING
# ============================================================


def load_normal_metadata(normal_dir: Path) -> pd.DataFrame:
    metadata_path = normal_dir / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Normal metadata not found: {metadata_path}. "
            "S2 parameters must come from the existing normal Q66 workload."
        )

    df = pd.read_csv(metadata_path)

    missing = [
        column
        for column in REQUIRED_METADATA_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Normal metadata is missing required columns: {missing}"
        )

    df = df.dropna(subset=REQUIRED_METADATA_COLUMNS).copy()
    if df.empty:
        raise ValueError("Normal Q66 metadata contains no usable rows.")

    for column in ["smc1", "smc2", "netone", "nettwo", "salesone", "salestwo"]:
        df[column] = df[column].astype(str)

    for column in ["year", "timeone", "time_window"]:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)

    # Remove duplicate SQL parameter combinations before sampling. This avoids
    # generating repeated structural queries even if metadata contains duplicate
    # normal executions.
    df = df.drop_duplicates(subset=REQUIRED_METADATA_COLUMNS).reset_index(drop=True)

    if len(df) < N_S2_PER_TYPE:
        raise ValueError(
            f"Only {len(df)} unique normal parameter combinations are available, "
            f"but N_S2_PER_TYPE={N_S2_PER_TYPE}."
        )

    print(f"[PARAMS] normal metadata: {metadata_path}")
    print(f"[PARAMS] unique usable combinations: {len(df):,}")

    return df


def row_to_params(row: pd.Series) -> dict:
    params = {
        "smc1": str(row["smc1"]),
        "smc2": str(row["smc2"]),
        "year": int(row["year"]),
        "timeone": int(row["timeone"]),
        "time_window": int(row["time_window"]),
        "netone": str(row["netone"]),
        "nettwo": str(row["nettwo"]),
        "salesone": str(row["salesone"]),
        "salestwo": str(row["salestwo"]),
    }

    if "file" in row.index and pd.notna(row["file"]):
        params["source_normal_file"] = str(row["file"])
    else:
        params["source_normal_file"] = ""

    return params


def sample_parameter_sets(
    normal_metadata: pd.DataFrame,
    rng: np.random.Generator,
    n: int,
) -> list[dict]:
    indices = rng.choice(
        normal_metadata.index.to_numpy(),
        size=n,
        replace=False,
    )

    return [
        row_to_params(normal_metadata.loc[int(index)])
        for index in indices
    ]


# ============================================================
# SQL HELPERS
# ============================================================


def sql_escape(value: str) -> str:
    return str(value).replace("'", "''")


def prepare_sql_params(params: dict) -> dict:
    prepared = dict(params)
    prepared["smc1"] = sql_escape(prepared["smc1"])
    prepared["smc2"] = sql_escape(prepared["smc2"])
    return prepared


def render_sql(template: str, params: dict) -> str:
    return template.format(**prepare_sql_params(params))


def included_branches() -> list[str]:
    # All retained main S2 mutations preserve the two-branch identity of Q66.
    return ["web", "catalog"]


def branch_config(branch: str) -> dict:
    if branch == "web":
        return {
            "sales_table": "web_sales",
            "sales_alias": "ws",
            "warehouse_key": "ws_warehouse_sk",
            "sold_date_key": "ws_sold_date_sk",
            "sold_time_key": "ws_sold_time_sk",
            "ship_mode_key": "ws_ship_mode_sk",
            "customer_key": "ws_bill_customer_sk",
            "item_key": "ws_item_sk",
            "quantity_col": "ws_quantity",
            "sales_param": "salesone",
            "net_param": "netone",
        }

    if branch == "catalog":
        return {
            "sales_table": "catalog_sales",
            "sales_alias": "cs",
            "warehouse_key": "cs_warehouse_sk",
            "sold_date_key": "cs_sold_date_sk",
            "sold_time_key": "cs_sold_time_sk",
            "ship_mode_key": "cs_ship_mode_sk",
            "customer_key": "cs_bill_customer_sk",
            "item_key": "cs_item_sk",
            "quantity_col": "cs_quantity",
            "sales_param": "salestwo",
            "net_param": "nettwo",
        }

    raise ValueError(f"Unknown Q66 branch: {branch}")


def use_customer_join(subtype: str) -> bool:
    return subtype == "q66_s2_join_add_customer"


def use_item_join(subtype: str) -> bool:
    return subtype == "q66_s2_join_add_item"


def include_time_predicate(subtype: str) -> bool:
    return subtype != "q66_s2_where_remove_time"


# ============================================================
# SQL BUILDER
# ============================================================


def build_branch_sql(
    branch: str,
    subtype: str,
    *,
    with_metrics: bool,
    with_prov: bool,
) -> str:
    cfg = branch_config(branch)

    table = cfg["sales_table"]
    sales_alias = cfg["sales_alias"]
    quantity_col = cfg["quantity_col"]
    sales_param = cfg["sales_param"]
    net_param = cfg["net_param"]

    select_columns = [
        "w.w_warehouse_name",
        "w.w_warehouse_sq_ft",
        "w.w_city",
        "w.w_county",
        "w.w_state",
        "w.w_country",
        "'{smc1}' || ',' || '{smc2}' AS ship_carriers",
        "d.d_year AS year",
    ]

    if with_metrics:
        for month_number, month_name in MONTHS:
            select_columns.append(
                f"sum(case when d.d_moy = {month_number} "
                f"then {sales_alias}.{{{sales_param}}} * "
                f"{sales_alias}.{quantity_col} else 0 end) "
                f"AS {month_name}_sales"
            )

        for month_number, month_name in MONTHS:
            select_columns.append(
                f"sum(case when d.d_moy = {month_number} "
                f"then {sales_alias}.{{{net_param}}} * "
                f"{sales_alias}.{quantity_col} else 0 end) "
                f"AS {month_name}_net"
            )

    joins = [
        f"FROM {table} {sales_alias}",
        (
            f"JOIN warehouse w ON "
            f"{sales_alias}.{cfg['warehouse_key']} = w.w_warehouse_sk"
        ),
        (
            f"JOIN date_dim d ON "
            f"{sales_alias}.{cfg['sold_date_key']} = d.d_date_sk"
        ),
        (
            f"JOIN time_dim t ON "
            f"{sales_alias}.{cfg['sold_time_key']} = t.t_time_sk"
        ),
        (
            f"JOIN ship_mode sm ON "
            f"{sales_alias}.{cfg['ship_mode_key']} = sm.sm_ship_mode_sk"
        ),
    ]

    # One witness corresponds to one contributing sales tuple and the dimension
    # tuples joined to it. The terms below mirror the actual relational joins.
    provenance_terms = [
        f"{sales_alias}.prov",
        "w.prov",
        "d.prov",
        "t.prov",
        "sm.prov",
    ]

    if use_customer_join(subtype):
        joins.append(
            f"JOIN customer c ON "
            f"{sales_alias}.{cfg['customer_key']} = c.c_customer_sk"
        )
        provenance_terms.append("c.prov")

    if use_item_join(subtype):
        joins.append(
            f"JOIN item i ON "
            f"{sales_alias}.{cfg['item_key']} = i.i_item_sk"
        )
        provenance_terms.append("i.prov")

    if with_prov:
        select_columns.append(
            "array_agg(list_value("
            + ", ".join(provenance_terms)
            + ")) AS prov"
        )

    where_conditions = [
        "d.d_year = {year}",
        "sm.sm_carrier IN ('{smc1}', '{smc2}')",
    ]

    if include_time_predicate(subtype):
        where_conditions.insert(
            1,
            "t.t_time BETWEEN {timeone} AND {timeone}+{time_window}",
        )

    group_by_columns = [
        "w.w_warehouse_name",
        "w.w_warehouse_sq_ft",
        "w.w_city",
        "w.w_county",
        "w.w_state",
        "w.w_country",
        "d.d_year",
    ]

    return (
        "SELECT\n    "
        + ",\n    ".join(select_columns)
        + "\n"
        + "\n".join(joins)
        + "\nWHERE "
        + "\n  AND ".join(where_conditions)
        + "\nGROUP BY\n    "
        + ",\n    ".join(group_by_columns)
    )


def build_base_cte(
    subtype: str,
    *,
    with_metrics: bool,
    with_prov: bool,
) -> str:
    branches = [
        build_branch_sql(
            branch,
            subtype,
            with_metrics=with_metrics,
            with_prov=with_prov,
        )
        for branch in included_branches()
    ]

    return "WITH base AS (\n" + "\n\nUNION ALL\n\n".join(branches) + "\n)"


def aggregate_metric_selects() -> list[str]:
    columns: list[str] = []

    for _, month_name in MONTHS:
        columns.append(f"sum({month_name}_sales) AS {month_name}_sales")

    for _, month_name in MONTHS:
        columns.append(
            f"sum({month_name}_sales / nullif(w_warehouse_sq_ft, 0)) "
            f"AS {month_name}_sales_per_sq_foot"
        )

    for _, month_name in MONTHS:
        columns.append(f"sum({month_name}_net) AS {month_name}_net")

    return columns


def final_limit_clause(subtype: str, *, materialized: bool) -> str:
    if not materialized:
        return ""

    if subtype == "q66_s2_limit_1":
        return "LIMIT 1"

    return "LIMIT 100"


def build_full_sql(subtype: str) -> str:
    cte = build_base_cte(
        subtype,
        with_metrics=True,
        with_prov=True,
    )

    metric_columns = aggregate_metric_selects()
    provenance_expression = (
        "list_reduce(array_agg(prov), "
        "(x, y) -> list_concat(x, y)) AS prov"
    )

    if subtype == "q66_s2_agg_group_by_state":
        dimensions = [
            "w_state",
            "w_country",
            "ship_carriers",
            "year",
        ]
        select_columns = dimensions + metric_columns + [provenance_expression]

        return (
            f"{cte}\n"
            "SELECT\n    "
            + ",\n    ".join(select_columns)
            + "\nFROM base\nGROUP BY\n    "
            + ",\n    ".join(dimensions)
            + "\nORDER BY w_state, w_country, ship_carriers, year\n"
            + "LIMIT 100"
        )

    dimensions = [
        "w_warehouse_name",
        "w_warehouse_sq_ft",
        "w_city",
        "w_county",
        "w_state",
        "w_country",
        "ship_carriers",
        "year",
    ]
    select_columns = dimensions + metric_columns + [provenance_expression]
    limit = final_limit_clause(subtype, materialized=True)

    return (
        f"{cte}\n"
        "SELECT\n    "
        + ",\n    ".join(select_columns)
        + "\nFROM base\nGROUP BY\n    "
        + ",\n    ".join(dimensions)
        + "\nORDER BY w_warehouse_name\n"
        + limit
    )


def build_group_key_sql(subtype: str, *, materialized: bool) -> str:
    cte = build_base_cte(
        subtype,
        with_metrics=False,
        with_prov=False,
    )

    if subtype == "q66_s2_agg_group_by_state":
        dimensions = [
            "w_state",
            "w_country",
            "ship_carriers",
            "year",
        ]
        limit = "LIMIT 100" if materialized else ""

        return (
            f"{cte}\n"
            "SELECT\n    "
            + ",\n    ".join(dimensions)
            + "\nFROM base\nGROUP BY\n    "
            + ",\n    ".join(dimensions)
            + "\nORDER BY w_state, w_country, ship_carriers, year\n"
            + limit
        )

    dimensions = [
        "w_warehouse_name",
        "w_warehouse_sq_ft",
        "w_city",
        "w_county",
        "w_state",
        "w_country",
        "ship_carriers",
        "year",
    ]
    limit = final_limit_clause(subtype, materialized=materialized)

    return (
        f"{cte}\n"
        "SELECT\n    "
        + ",\n    ".join(dimensions)
        + "\nFROM base\nGROUP BY\n    "
        + ",\n    ".join(dimensions)
        + "\nORDER BY w_warehouse_name\n"
        + limit
    )


def build_count_sql(subtype: str, *, materialized: bool) -> str:
    group_query = build_group_key_sql(
        subtype,
        materialized=materialized,
    )
    return f"SELECT count(*) AS n_rows FROM (\n{group_query}\n) q"


# ============================================================
# PROVENANCE VALIDATION
# ============================================================


def as_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def list_like(value: Any) -> list:
    value = as_python(value)

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)

    return [value]


def validate_full_provenance_result(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    params: dict,
) -> dict:
    """
    Execute one complete mutated query and verify the expected Q66 provenance
    shape. Every top-level element in result.prov must be one witness, and each
    witness must contain exactly one token for each table participating in the
    corresponding branch join.
    """

    sql = render_sql(build_full_sql(subtype), params)
    result = con.execute(sql).df()

    if result.empty:
        raise ValueError(f"{subtype}: full query returned no rows")

    if "prov" not in result.columns:
        raise ValueError(f"{subtype}: full query has no prov column")

    expected_size = int(S2_INFO[subtype]["expected_witness_size"])
    witness_count = 0
    observed_sizes: set[int] = set()

    for cell in result["prov"].tolist():
        witnesses = list_like(cell)

        for witness in witnesses:
            witness_values = list_like(witness)
            if not witness_values:
                continue

            witness_count += 1
            observed_sizes.add(len(witness_values))

    if witness_count == 0:
        raise ValueError(f"{subtype}: no non-empty provenance witnesses")

    if observed_sizes != {expected_size}:
        raise ValueError(
            f"{subtype}: expected witness size {expected_size}, "
            f"observed sizes={sorted(observed_sizes)}"
        )

    return {
        "full_validation_nrows": int(len(result)),
        "full_validation_witnesses": int(witness_count),
        "full_validation_witness_sizes": ",".join(
            str(size) for size in sorted(observed_sizes)
        ),
    }


# ============================================================
# QUERY VALIDATION / WRITING
# ============================================================


def count_rows(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    params: dict,
    *,
    materialized: bool,
) -> int:
    sql = render_sql(
        build_count_sql(subtype, materialized=materialized),
        params,
    )
    row = con.execute(sql).fetchone()

    if row is None:
        return 0

    return int(row[0])


def validate_instance(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    params: dict,
) -> tuple[bool, dict]:
    materialized_nrows = count_rows(
        con,
        subtype,
        params,
        materialized=True,
    )
    unbounded_nrows = count_rows(
        con,
        subtype,
        params,
        materialized=False,
    )

    valid = materialized_nrows > 0 and unbounded_nrows > 0

    return valid, {
        "n_rows": materialized_nrows,
        "limited_nrows": materialized_nrows,
        "unbounded_nrows": unbounded_nrows,
    }


def write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No metadata rows were generated.")

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_directory() -> None:
    if CLEAN_OUTPUT_DIR and OUT_S2.exists():
        print(f"[CLEAN] removing {OUT_S2}")
        shutil.rmtree(OUT_S2)

    OUT_S2.mkdir(parents=True, exist_ok=True)


def subtype_filename(subtype: str, index: int) -> str:
    short_name = subtype.removeprefix("q66_s2_")
    return f"s2_{short_name}_{index:05d}.sql"


def generate_subtype(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    parameter_sets: list[dict],
    global_rows: list[dict],
) -> None:
    info = S2_INFO[subtype]
    full_sql_template = build_full_sql(subtype)
    full_checks_done = 0

    print(f"\n[GEN] {subtype}")
    print(f"      target={len(parameter_sets)}")

    for index, base_params in enumerate(parameter_sets):
        params = dict(base_params)

        valid, count_metadata = validate_instance(
            con,
            subtype,
            params,
        )
        if not valid:
            raise RuntimeError(
                f"{subtype}: sampled normal parameter set produced an empty "
                f"mutated query: {params}"
            )

        provenance_validation = {
            "full_validation_nrows": "",
            "full_validation_witnesses": "",
            "full_validation_witness_sizes": "",
        }

        if (
            VALIDATE_FULL_PROVENANCE
            and full_checks_done < FULL_PROVENANCE_CHECKS_PER_SUBTYPE
        ):
            provenance_validation = validate_full_provenance_result(
                con,
                subtype,
                params,
            )
            full_checks_done += 1

            print(
                f"  [PROV OK] {subtype}: "
                f"rows={provenance_validation['full_validation_nrows']} "
                f"witnesses={provenance_validation['full_validation_witnesses']} "
                f"witness_size={provenance_validation['full_validation_witness_sizes']}"
            )

        sql = render_sql(full_sql_template, params).strip() + "\n"
        filename = subtype_filename(subtype, index)
        output_path = OUT_S2 / filename
        output_path.write_text(sql, encoding="utf-8")

        global_rows.append(
            {
                "file": filename,
                "label": int(info["label"]),
                "scenario": subtype,
                "anomaly_class": info["anomaly_class"],
                "s2_category": info["category"],
                "logical_unit": info["logical_unit"],
                "intent_preservation": info["intent_preservation"],
                "description": info["description"],
                "expected_witness_size": int(info["expected_witness_size"]),
                **params,
                **count_metadata,
                **provenance_validation,
            }
        )

        completed = index + 1
        if completed % 100 == 0 or completed == len(parameter_sets):
            print(f"  [GEN] {subtype}: {completed}/{len(parameter_sets)}")

    print(f"  [OK] {subtype}: {len(parameter_sets)} queries")


# ============================================================
# FINAL DIAGNOSTICS
# ============================================================


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print("[Q66 S2 GENERATION SUMMARY]")
    print("=" * 90)
    print(f"total queries={len(df):,}")
    print(f"scenarios={df['scenario'].value_counts().to_dict()}")
    print(f"labels={df['label'].value_counts().to_dict()}")
    print(f"anomaly classes={df['anomaly_class'].value_counts().to_dict()}")

    for subtype in S2_SUBTYPES:
        group = df[df["scenario"] == subtype]
        if group.empty:
            continue

        print("\n" + "-" * 90)
        print(f"[{subtype}] n={len(group):,}")
        print(
            f"  n_rows: min={int(group['n_rows'].min())} "
            f"median={float(group['n_rows'].median()):.1f} "
            f"max={int(group['n_rows'].max())}"
        )
        print(
            f"  unbounded_nrows: min={int(group['unbounded_nrows'].min())} "
            f"median={float(group['unbounded_nrows'].median()):.1f} "
            f"max={int(group['unbounded_nrows'].max())}"
        )
        print(
            f"  carriers={len(set(group['smc1']) | set(group['smc2']))} "
            f"timeone=[{int(group['timeone'].min())}, {int(group['timeone'].max())}]"
        )
        print(
            f"  expected witness size={int(group['expected_witness_size'].iloc[0])}"
        )

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q66 S2 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q66.")
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
    for name in ['q66_s2_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, EXTERNAL_BASE_DIR, N_S2_PER_TYPE, OUT_S2, Q66_NORMAL_DIR, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_S2_PER_TYPE = args.n_per_type
    OUT_S2 = args.output_dir / "q66_s2_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir
    Q66_NORMAL_DIR = args.normal_dir


def main() -> None:
    configure_cli()
    print("=" * 90)
    print("[Q66 FINAL S2 GENERATOR]")
    print("=" * 90)
    print(f"DB_PATH={DB_PATH}")
    print(f"DB_SCHEMA={DB_SCHEMA}")
    print(f"Q66_NORMAL_DIR={Q66_NORMAL_DIR}")
    print(f"OUT_S2={OUT_S2}")
    print(f"N_S2_PER_TYPE={N_S2_PER_TYPE}")
    print(f"MATCH_PARAMETERS_ACROSS_SUBTYPES={MATCH_PARAMETERS_ACROSS_SUBTYPES}")
    print(f"VALIDATE_FULL_PROVENANCE={VALIDATE_FULL_PROVENANCE}")
    print(f"FULL_PROVENANCE_CHECKS_PER_SUBTYPE={FULL_PROVENANCE_CHECKS_PER_SUBTYPE}")
    print(f"S2_SUBTYPES={S2_SUBTYPES}")
    print("=" * 90)

    if set(S2_SUBTYPES) != set(S2_INFO):
        raise ValueError("S2_SUBTYPES and S2_INFO are inconsistent.")

    prepare_output_directory()

    normal_metadata = load_normal_metadata(Q66_NORMAL_DIR)
    rng = np.random.default_rng(SEED)

    matched_parameter_sets: Optional[list[dict]] = None
    if MATCH_PARAMETERS_ACROSS_SUBTYPES:
        matched_parameter_sets = sample_parameter_sets(
            normal_metadata,
            rng,
            N_S2_PER_TYPE,
        )

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET search_path = ?", [DB_SCHEMA])

    all_rows: list[dict] = []

    try:
        for subtype in S2_SUBTYPES:
            if matched_parameter_sets is not None:
                parameter_sets = [
                    dict(params)
                    for params in matched_parameter_sets
                ]
            else:
                parameter_sets = sample_parameter_sets(
                    normal_metadata,
                    rng,
                    N_S2_PER_TYPE,
                )

            generate_subtype(
                con,
                subtype,
                parameter_sets,
                all_rows,
            )
    finally:
        con.close()

    metadata_path = OUT_S2 / "metadata.csv"
    write_metadata(metadata_path, all_rows)
    print_summary(all_rows)

    print("\n[DONE]")
    print(f"  queries   : {len(all_rows):,}")
    print(f"  directory : {OUT_S2}")
    print(f"  metadata  : {metadata_path}")


if __name__ == "__main__":
    main()

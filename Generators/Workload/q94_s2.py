#!/usr/bin/env python3
from __future__ import annotations

import argparse

import csv
import re
from pathlib import Path
from typing import Any, Dict

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

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

Q94_NORMAL_DIR = None  # Set by command-line arguments.
OUT_S2 = None  # Set by command-line arguments.

N_PROV_SHAPE_CHECKS_PER_SUBTYPE = 3
MAX_FACTORS_CHECKED_PER_QUERY = 100

S2_SUBTYPES = [
    "q94_s2_join_add_item",
    "q94_s2_join_add_customer",
    "q94_s2_join_add_warehouse",
    "q94_s2_where_remove_company",
    "q94_s2_where_remove_returns_exclusion",
    "q94_s2_agg_nested_by_order",
]

S2_INFO: Dict[str, dict] = {
    "q94_s2_join_add_item": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_outer_witness_size": 5,
        "expected_nested_witness_size": 1,
        "expected_expanded_witness_size": 6,
        "description": (
            "Adds item through ws1_item_sk = i_item_sk and appends item "
            "provenance to the outer witness factor."
        ),
    },
    "q94_s2_join_add_customer": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_outer_witness_size": 5,
        "expected_nested_witness_size": 1,
        "expected_expanded_witness_size": 6,
        "description": (
            "Adds the billed customer through ws1_bill_customer_sk = "
            "c_customer_sk and appends customer provenance to the outer "
            "witness factor."
        ),
    },
    "q94_s2_join_add_warehouse": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_outer_witness_size": 5,
        "expected_nested_witness_size": 1,
        "expected_expanded_witness_size": 6,
        "description": (
            "Adds the outer warehouse through ws1_warehouse_sk = "
            "w_warehouse_sk and appends warehouse provenance to the outer "
            "witness factor."
        ),
    },
    "q94_s2_where_remove_company": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "expected_outer_witness_size": 4,
        "expected_nested_witness_size": 1,
        "expected_expanded_witness_size": 5,
        "description": (
            "Removes web_company_name = 'pri' while retaining web_site, its "
            "join, and its provenance."
        ),
    },
    "q94_s2_where_remove_returns_exclusion": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "anti-join predicate removal",
        "expected_outer_witness_size": 4,
        "expected_nested_witness_size": 1,
        "expected_expanded_witness_size": 5,
        "description": (
            "Removes the NOT EXISTS condition over web_returns and admits "
            "multi-warehouse orders that have associated returns."
        ),
    },
    "q94_s2_agg_nested_by_order": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_aggregation",
        "logical_unit": "nested aggregation-granularity change",
        "expected_outer_witness_size": 4,
        "expected_nested_witness_size": 1,
        "expected_expanded_witness_size": 5,
        "description": (
            "Aggregates nested provenance by order rather than by order and "
            "target warehouse while preserving the cross-warehouse "
            "inequality."
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

    metadata_path = Q94_NORMAL_DIR / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Normal Q94 metadata does not exist: {metadata_path}"
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
    metadata_path = Q94_NORMAL_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)

    required = [
        "file",
        "state",
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
            f"Normal Q94 metadata is missing columns: {missing}"
        )

    df = df.dropna(subset=required).copy()

    if df.empty:
        raise ValueError("Normal Q94 metadata contains no usable rows.")

    df["file"] = df["file"].astype(str)
    df["state"] = df["state"].astype(str)
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
        "state": sql_escape(str(row["state"])),
        "sdate": str(row["sdate"]),
        "window_days": int(row["window_days"]),
    }


def subtype_spec(subtype: str) -> dict:
    return {
        "add_item": subtype == "q94_s2_join_add_item",
        "add_customer": subtype == "q94_s2_join_add_customer",
        "add_warehouse": subtype == "q94_s2_join_add_warehouse",
        "remove_company": subtype == "q94_s2_where_remove_company",
        "remove_returns": (
            subtype == "q94_s2_where_remove_returns_exclusion"
        ),
        "nested_by_order": (
            subtype == "q94_s2_agg_nested_by_order"
        ),
    }


def build_ctes(
    subtype: str,
    with_provenance: bool,
) -> str:
    spec = subtype_spec(subtype)

    if spec["nested_by_order"]:
        nested_key_select = (
            "ow.ws_order_number as ws_order_number"
        )
        nested_group_by = ["ow.ws_order_number"]
        nested_outer_join = (
            "nested.ws_order_number = ws1.ws_order_number"
        )
    else:
        nested_key_select = (
            "ow.ws_order_number as ws_order_number,\n"
            "        ow.ws_warehouse_sk as ws_warehouse_sk"
        )
        nested_group_by = [
            "ow.ws_order_number",
            "ow.ws_warehouse_sk",
        ]
        nested_outer_join = (
            "nested.ws_order_number = ws1.ws_order_number\n"
            " and nested.ws_warehouse_sk = ws1.ws_warehouse_sk"
        )

    nested_group_sql = ",\n        ".join(
        nested_group_by
    )

    nested_prov_select = (
        ",\n        array_agg(ws2.prov) as nested_prov"
        if with_provenance
        else ""
    )

    outer_from_lines = [
        "web_sales ws1",
        "join nested",
        f"  on {nested_outer_join}",
        "join date_dim d",
        "  on ws1.ws_ship_date_sk = d.d_date_sk",
        "join customer_address ca",
        "  on ws1.ws_ship_addr_sk = ca.ca_address_sk",
        "join web_site web",
        "  on ws1.ws_web_site_sk = web.web_site_sk",
    ]

    outer_provenance = [
        "ws1.prov",
        "d.prov",
        "ca.prov",
        "web.prov",
    ]

    if spec["add_item"]:
        outer_from_lines.extend(
            [
                "join item i",
                "  on ws1.ws_item_sk = i.i_item_sk",
            ]
        )
        outer_provenance.append("i.prov")

    if spec["add_customer"]:
        outer_from_lines.extend(
            [
                "join customer c",
                "  on ws1.ws_bill_customer_sk = c.c_customer_sk",
            ]
        )
        outer_provenance.append("c.prov")

    if spec["add_warehouse"]:
        outer_from_lines.extend(
            [
                "join warehouse w",
                "  on ws1.ws_warehouse_sk = w.w_warehouse_sk",
            ]
        )
        outer_provenance.append("w.prov")

    outer_from_sql = "\n    ".join(
        outer_from_lines
    )

    outer_where_conditions = [
        (
            "d.d_date between cast('{sdate}' as date) "
            "and cast('{sdate}' as date) "
            "+ interval {window_days} day"
        ),
        "ca.ca_state = '{state}'",
    ]

    if not spec["remove_company"]:
        outer_where_conditions.append(
            "web.web_company_name = 'pri'"
        )

    if not spec["remove_returns"]:
        outer_where_conditions.append(
            "not exists (\n"
            "        select 1\n"
            "        from web_returns wr1\n"
            "        where wr1.wr_order_number = ws1.ws_order_number\n"
            "      )"
        )

    outer_where_sql = "\n      and ".join(
        outer_where_conditions
    )

    if with_provenance:
        singleton_factors = ",\n            ".join(
            f"list_value({token})"
            for token in outer_provenance
        )

        qualified_prov_select = (
            ",\n        list_value(\n"
            f"            {singleton_factors},\n"
            "            nested.nested_prov\n"
            "        ) as factor_prov"
        )
    else:
        qualified_prov_select = ""

    return f"""
with order_warehouses as (
    select distinct
        ws_order_number,
        ws_warehouse_sk
    from web_sales
),
nested as (
    select
        {nested_key_select}
        {nested_prov_select}
    from web_sales ws2
    join order_warehouses ow
      on ws2.ws_order_number = ow.ws_order_number
     and ws2.ws_warehouse_sk <> ow.ws_warehouse_sk
    group by
        {nested_group_sql}
),
qualified as (
    select
        ws1.ws_order_number,
        ws1.ws_ext_ship_cost,
        ws1.ws_net_profit
        {qualified_prov_select}
    from {outer_from_sql}
    where {outer_where_sql}
)
"""


def build_final_sql(
    subtype: str,
    with_provenance: bool,
) -> str:
    ctes = build_ctes(
        subtype=subtype,
        with_provenance=with_provenance,
    )

    prov_select = (
        ",\n    array_agg(factor_prov) as prov"
        if with_provenance
        else ""
    )

    return (
        ctes
        + f"""
select
    count(
        distinct ws_order_number
    ) as "order count",
    sum(ws_ext_ship_cost) as "total shipping cost",
    sum(ws_net_profit) as "total net profit"
    {prov_select}
from qualified
order by
    count(distinct ws_order_number)
limit 100
"""
    )


def build_contributor_count_sql(
    subtype: str,
) -> str:
    return (
        build_ctes(
            subtype=subtype,
            with_provenance=False,
        )
        + """
select count(*) as n_rows
from qualified
"""
    )


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

    # Both S1 parameters remain in every retained mutation.
    require(
        "ca.ca_state =" in lower,
        f"{subtype}: sampled state predicate is missing.",
    )
    require(
        "d.d_date between" in lower,
        f"{subtype}: sampled date-window predicate is missing.",
    )

    require(
        "ws2.ws_order_number = ow.ws_order_number" in lower,
        f"{subtype}: nested order match is missing.",
    )
    require(
        "ws2.ws_warehouse_sk <> ow.ws_warehouse_sk" in lower,
        f"{subtype}: cross-warehouse inequality is missing.",
    )
    require(
        "nested.ws_order_number = ws1.ws_order_number" in lower,
        f"{subtype}: outer nested-order join is missing.",
    )
    require(
        "ws1.ws_ship_date_sk = d.d_date_sk" in lower,
        f"{subtype}: shipping-date join is missing.",
    )
    require(
        "ws1.ws_ship_addr_sk = ca.ca_address_sk" in lower,
        f"{subtype}: shipping-address join is missing.",
    )
    require(
        "ws1.ws_web_site_sk = web.web_site_sk" in lower,
        f"{subtype}: web-site join is missing.",
    )

    if subtype == "q94_s2_join_add_item":
        require(
            "join item i" in lower,
            "join_add_item: item relation is missing.",
        )
        require(
            "ws1.ws_item_sk = i.i_item_sk" in lower,
            "join_add_item: item join is missing.",
        )
        require(
            has_alias_prov(lower, "i"),
            "join_add_item: item provenance is missing.",
        )

    elif subtype == "q94_s2_join_add_customer":
        require(
            "join customer c" in lower,
            "join_add_customer: customer relation is missing.",
        )
        require(
            "ws1.ws_bill_customer_sk = c.c_customer_sk" in lower,
            "join_add_customer: customer join is missing.",
        )
        require(
            has_alias_prov(lower, "c"),
            "join_add_customer: customer provenance is missing.",
        )

    elif subtype == "q94_s2_join_add_warehouse":
        require(
            "join warehouse w" in lower,
            "join_add_warehouse: warehouse relation is missing.",
        )
        require(
            "ws1.ws_warehouse_sk = w.w_warehouse_sk" in lower,
            "join_add_warehouse: warehouse join is missing.",
        )
        require(
            has_alias_prov(lower, "w"),
            "join_add_warehouse: warehouse provenance is missing.",
        )

    elif subtype == "q94_s2_where_remove_company":
        require(
            "web.web_company_name =" not in lower,
            "where_remove_company: company predicate remains.",
        )
        require(
            has_alias_prov(lower, "web"),
            "where_remove_company: web-site provenance is missing.",
        )

    elif subtype == "q94_s2_where_remove_returns_exclusion":
        require(
            "not exists" not in lower,
            "where_remove_returns_exclusion: NOT EXISTS remains.",
        )

    elif subtype == "q94_s2_agg_nested_by_order":
        require(
            "group by ow.ws_order_number" in lower,
            "agg_nested_by_order: order-level GROUP BY is missing.",
        )
        require(
            "ow.ws_warehouse_sk as ws_warehouse_sk" not in lower,
            "agg_nested_by_order: target warehouse remains projected.",
        )
        require(
            "nested.ws_warehouse_sk = ws1.ws_warehouse_sk"
            not in lower,
            "agg_nested_by_order: target-warehouse outer join remains.",
        )

    if subtype != "q94_s2_where_remove_company":
        require(
            "web.web_company_name = 'pri'" in lower,
            f"{subtype}: company predicate is missing.",
        )

    if subtype != "q94_s2_where_remove_returns_exclusion":
        require(
            "not exists" in lower,
            f"{subtype}: return-exclusion predicate is missing.",
        )


# ============================================================
# EXECUTION AND FACTORISED-PROVENANCE VALIDATION
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
) -> int:
    sql = render_sql(
        build_final_sql(
            subtype=subtype,
            with_provenance=False,
        ),
        row,
    ).strip().rstrip(";")

    return execute_scalar_count(
        con,
        f"""
select count(*)
from (
{sql}
) q94_result_count
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


def parse_factor(
    factor: Any,
) -> tuple[list[Any], list[list[Any]]]:
    """
    Factor structure:
        [
            [outer token 1],
            ...,
            [outer token n],
            nested_prov
        ]
    """

    factor = as_python(factor)

    if not isinstance(factor, list) or len(factor) < 2:
        raise RuntimeError(
            f"Unexpected Q94 factor shape: {factor!r}"
        )

    singleton_components = factor[:-1]
    nested_collection = as_python(factor[-1])

    outer_tokens: list[Any] = []

    for component in singleton_components:
        outer_tokens.extend(
            flatten_scalar_tokens(component)
        )

    if nested_collection is None:
        nested_witnesses: list[list[Any]] = []
    else:
        if not isinstance(nested_collection, list):
            nested_collection = [nested_collection]

        nested_witnesses = [
            flatten_scalar_tokens(witness)
            for witness in nested_collection
            if witness is not None
        ]

    return outer_tokens, nested_witnesses


def validate_full_provenance(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    row: pd.Series,
) -> None:
    sql = render_sql(
        build_final_sql(
            subtype=subtype,
            with_provenance=True,
        ),
        row,
    )

    df = con.execute(sql).fetchdf()

    if df.empty:
        raise RuntimeError(
            f"{subtype}: full provenance query returned no result row."
        )

    prov_cell = as_python(df["prov"].iloc[0])

    if prov_cell is None or not isinstance(prov_cell, list):
        raise RuntimeError(
            f"{subtype}: no factorised provenance was produced."
        )

    info = S2_INFO[subtype]
    expected_outer = int(
        info["expected_outer_witness_size"]
    )
    expected_nested = int(
        info["expected_nested_witness_size"]
    )
    expected_expanded = int(
        info["expected_expanded_witness_size"]
    )

    checked = 0

    for factor in prov_cell:
        outer_tokens, nested_witnesses = parse_factor(
            factor
        )

        if len(outer_tokens) != expected_outer:
            raise RuntimeError(
                f"{subtype}: expected outer witness size "
                f"{expected_outer}, observed {len(outer_tokens)}."
            )

        if not nested_witnesses:
            raise RuntimeError(
                f"{subtype}: factor has no nested witnesses."
            )

        for nested_witness in nested_witnesses:
            if len(nested_witness) != expected_nested:
                raise RuntimeError(
                    f"{subtype}: expected nested witness size "
                    f"{expected_nested}, observed "
                    f"{len(nested_witness)}."
                )

            expanded_size = (
                len(outer_tokens) + len(nested_witness)
            )

            if expanded_size != expected_expanded:
                raise RuntimeError(
                    f"{subtype}: expected expanded witness size "
                    f"{expected_expanded}, observed {expanded_size}."
                )

        checked += 1

        if checked >= MAX_FACTORS_CHECKED_PER_QUERY:
            break

    if checked == 0:
        raise RuntimeError(
            f"{subtype}: no factorised provenance factors were checked."
        )

    print(
        f"[PROV OK] {subtype} "
        f"source={row['file']} "
        f"factors={checked} "
        f"outer={expected_outer} "
        f"nested={expected_nested} "
        f"expanded={expected_expanded}"
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
                build_final_sql(
                    subtype=subtype,
                    with_provenance=True,
                ),
                row,
            )

            validate_structure(subtype, sql)

            n_contributors = contributor_count(
                con,
                subtype,
                row,
            )

            if n_contributors <= 0:
                raise RuntimeError(
                    f"{subtype}: source {row['file']} produced no "
                    "qualified tuples during pre-check."
                )

            n_result_rows = result_count(
                con,
                subtype,
                row,
            )

            if n_result_rows != 1:
                raise RuntimeError(
                    f"{subtype}: expected one final aggregate row, "
                    f"observed {n_result_rows}."
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
    short_name = subtype.replace("q94_s2_", "")

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
            build_final_sql(
                subtype=subtype,
                with_provenance=True,
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

        if n_contributors <= 0:
            raise RuntimeError(
                f"{subtype}: preselected source {row['file']} produced "
                "no qualified tuples. The run is aborted and the source "
                "is not replaced."
            )

        n_result_rows = result_count(
            con,
            subtype,
            row,
        )

        if n_result_rows != 1:
            raise RuntimeError(
                f"{subtype}: expected one aggregate result row, "
                f"observed {n_result_rows}."
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
                "state": str(row["state"]),
                "sdate": str(row["sdate"]),
                "window_days": int(row["window_days"]),
                # Matches the normal Q94 metadata semantics.
                "n_rows": int(n_contributors),
                "result_n_rows": int(n_result_rows),
                "normal_n_rows": int(row["n_rows"]),
                "provenance_semantics": (
                    "factorized_multiwarehouse_order"
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
    p = argparse.ArgumentParser(description="Generate Q94 S2 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q94.")
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
    for name in ['q94_s2_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, EXTERNAL_BASE_DIR, N_S2_PER_TYPE, OUT_S2, Q94_NORMAL_DIR, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_S2_PER_TYPE = args.n_per_type
    EXTERNAL_BASE_DIR = args.output_dir
    Q94_NORMAL_DIR = args.normal_dir
    OUT_S2 = args.output_dir / "q94_s2_anomalies"


def main() -> None:
    configure_cli()
    print("=" * 96)
    print("[Q94 FINAL S2 GENERATOR]")
    print("=" * 96)
    print(f"DB_PATH={DB_PATH}")
    print(f"DB_SCHEMA={DB_SCHEMA}")
    print(f"Q94_NORMAL_DIR={Q94_NORMAL_DIR}")
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

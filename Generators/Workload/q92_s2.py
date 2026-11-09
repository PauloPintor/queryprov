#!/usr/bin/env python3
from __future__ import annotations

import argparse

"""
q92_s2_generator.py

Final S2 structural-anomaly generator for TPC-DS Q92.

Main S2 subtypes
----------------
- q92_s2_join_add_web_page
- q92_s2_join_add_customer
- q92_s2_where_remove_outer_date
- q92_s2_where_remove_discount_threshold
- q92_s2_agg_group_by_manufacturer

Methodological guarantees
-------------------------
- The same deterministic sample of already-generated normal Q92 executions is
  used for every S2 subtype.
- Source executions are sampled without conditioning on result cardinality,
  provenance size, or mutation magnitude.
- No failed, empty, or weak source execution is silently replaced.
- The normally sampled manufacturer and date are preserved in every mutation.
- The partial date-predicate removal keeps the sampled date in the threshold
  computation.
- Full factorised-provenance shape is checked on a deterministic prefix before
  bulk generation.
- Any invalid or empty preselected instance aborts the run rather than
  triggering post-hoc resampling.

Expected factorised provenance shapes
-------------------------------------
Normal Q92:
    outer witness   = [web_sales, item, outer date]
    average witness = [web_sales, average date]
    expanded size   = 5

Join additions:
    outer witness size   = 4
    average witness size = 2
    expanded size        = 6

Predicate removals:
    outer witness size   = 3
    average witness size = 2
    expanded size        = 5

Manufacturer-level threshold:
    outer witness size   = 3
    average witness size = 3
    expanded size        = 6
"""

import csv
import os
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

Q92_NORMAL_DIR = None  # Set by command-line arguments.
OUT_S2 = None  # Set by command-line arguments.

N_PROV_SHAPE_CHECKS_PER_SUBTYPE = 3
MAX_FACTORS_CHECKED_PER_QUERY = 100

S2_SUBTYPES = [
    "q92_s2_join_add_web_page",
    "q92_s2_join_add_customer",
    "q92_s2_where_remove_outer_date",
    "q92_s2_where_remove_discount_threshold",
    "q92_s2_agg_group_by_manufacturer",
]

S2_INFO: Dict[str, dict] = {
    "q92_s2_join_add_web_page": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_outer_witness_size": 4,
        "expected_avg_witness_size": 2,
        "expected_expanded_witness_size": 6,
        "description": (
            "Adds web_page through ws_web_page_sk = wp_web_page_sk and "
            "appends web-page provenance to the outer witness factor."
        ),
    },
    "q92_s2_join_add_customer": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_outer_witness_size": 4,
        "expected_avg_witness_size": 2,
        "expected_expanded_witness_size": 6,
        "description": (
            "Adds the billed customer through ws_bill_customer_sk = "
            "c_customer_sk and appends customer provenance to the outer "
            "witness factor."
        ),
    },
    "q92_s2_where_remove_outer_date": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "expected_outer_witness_size": 3,
        "expected_avg_witness_size": 2,
        "expected_expanded_witness_size": 5,
        "description": (
            "Removes the date-window predicate from the outer sale selection "
            "while preserving the normally sampled date in the average "
            "threshold computation."
        ),
    },
    "q92_s2_where_remove_discount_threshold": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "expected_outer_witness_size": 3,
        "expected_avg_witness_size": 2,
        "expected_expanded_witness_size": 5,
        "description": (
            "Removes ws_ext_discount_amt > discount_threshold while retaining "
            "the factorised average relation and both normally sampled S1 "
            "parameters."
        ),
    },
    "q92_s2_agg_group_by_manufacturer": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_aggregation",
        "logical_unit": "aggregation-granularity change",
        "expected_outer_witness_size": 3,
        "expected_avg_witness_size": 3,
        "expected_expanded_witness_size": 6,
        "description": (
            "Replaces the per-item average-discount threshold with one "
            "threshold per manufacturer and includes threshold-side item "
            "provenance."
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
            "Set DUCKDB_PATH to the correct path."
        )

    metadata_path = Q92_NORMAL_DIR / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Normal Q92 metadata does not exist: {metadata_path}"
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
    metadata_path = Q92_NORMAL_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)

    required = [
        "file",
        "manufact",
        "wsdate",
        "date_window_days",
        "n_rows",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Normal Q92 metadata is missing columns: {missing}"
        )

    df = df.dropna(subset=required).copy()

    if df.empty:
        raise ValueError("Normal Q92 metadata contains no usable rows.")

    df["file"] = df["file"].astype(str)
    df["manufact"] = pd.to_numeric(
        df["manufact"],
        errors="raise",
    ).astype(int)
    df["wsdate"] = pd.to_datetime(
        df["wsdate"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    df["date_window_days"] = pd.to_numeric(
        df["date_window_days"],
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
# SQL RENDERING
# ============================================================

def prepare_params(row: pd.Series | dict) -> dict:
    return {
        "manufact": int(row["manufact"]),
        "wsdate": str(row["wsdate"]),
        "date_window_days": int(row["date_window_days"]),
    }


def subtype_spec(subtype: str) -> dict:
    add_web_page = subtype == "q92_s2_join_add_web_page"
    add_customer = subtype == "q92_s2_join_add_customer"
    remove_outer_date = (
        subtype == "q92_s2_where_remove_outer_date"
    )
    remove_threshold = (
        subtype == "q92_s2_where_remove_discount_threshold"
    )
    group_by_manufacturer = (
        subtype == "q92_s2_agg_group_by_manufacturer"
    )

    outer_provenance = [
        "ws.prov",
        "i.prov",
        "d.prov",
    ]

    if add_web_page:
        outer_provenance.append("wp.prov")

    if add_customer:
        outer_provenance.append("c.prov")

    if group_by_manufacturer:
        average_provenance = [
            "ws_avg.prov",
            "d_avg.prov",
            "i_avg.prov",
        ]
    else:
        average_provenance = [
            "ws_avg.prov",
            "d_avg.prov",
        ]

    return {
        "add_web_page": add_web_page,
        "add_customer": add_customer,
        "remove_outer_date": remove_outer_date,
        "remove_threshold": remove_threshold,
        "group_by_manufacturer": group_by_manufacturer,
        "outer_provenance": outer_provenance,
        "average_provenance": average_provenance,
    }


def build_ctes(
    subtype: str,
    with_provenance: bool,
) -> str:
    spec = subtype_spec(subtype)

    avg_from_lines = [
        "web_sales ws_avg",
        "join date_dim d_avg",
        "  on d_avg.d_date_sk = ws_avg.ws_sold_date_sk",
    ]

    if spec["group_by_manufacturer"]:
        avg_from_lines.extend(
            [
                "join item i_avg",
                "  on i_avg.i_item_sk = ws_avg.ws_item_sk",
            ]
        )

    avg_from_sql = "\n    ".join(avg_from_lines)

    avg_where_sql = """
    where d_avg.d_date between cast('{wsdate}' as date)
                           and cast('{wsdate}' as date)
                               + interval {date_window_days} day
"""

    if spec["group_by_manufacturer"]:
        threshold_key_select = (
            "i_avg.i_manufact_id as threshold_key,\n        "
        )
        group_by_sql = """
    group by
        i_avg.i_manufact_id
"""
        threshold_join = (
            "ad.threshold_key = i.i_manufact_id"
        )
    else:
        threshold_key_select = (
            "ws_avg.ws_item_sk as threshold_key,\n        "
        )
        group_by_sql = """
    group by
        ws_avg.ws_item_sk
"""
        threshold_join = (
            "ad.threshold_key = i.i_item_sk"
        )

    avg_prov_sql = ""

    if with_provenance:
        avg_args = ",\n                    ".join(
            spec["average_provenance"]
        )
        avg_prov_sql = (
            ",\n        array_agg(\n"
            "            list_value(\n"
            f"                    {avg_args}\n"
            "            )\n"
            "        ) as avg_prov"
        )

    outer_from_lines = [
        "web_sales ws",
        "join item i",
        "  on i.i_item_sk = ws.ws_item_sk",
        "join date_dim d",
        "  on d.d_date_sk = ws.ws_sold_date_sk",
    ]

    if spec["add_web_page"]:
        outer_from_lines.extend(
            [
                "join web_page wp",
                "  on ws.ws_web_page_sk = wp.wp_web_page_sk",
            ]
        )

    if spec["add_customer"]:
        outer_from_lines.extend(
            [
                "join customer c",
                "  on ws.ws_bill_customer_sk = c.c_customer_sk",
            ]
        )

    outer_from_lines.extend(
        [
            "join avg_discount ad",
            f"  on {threshold_join}",
        ]
    )

    outer_from_sql = "\n    ".join(outer_from_lines)

    outer_where_conditions = [
        "i.i_manufact_id = {manufact}",
    ]

    if not spec["remove_outer_date"]:
        outer_where_conditions.append(
            "d.d_date between cast('{wsdate}' as date) "
            "and cast('{wsdate}' as date) "
            "+ interval {date_window_days} day"
        )

    if not spec["remove_threshold"]:
        outer_where_conditions.append(
            "ws.ws_ext_discount_amt > ad.discount_threshold"
        )

    outer_where_sql = "\n      and ".join(
        outer_where_conditions
    )

    if with_provenance:
        outer_args = ",\n                ".join(
            spec["outer_provenance"]
        )
        qualified_select = (
            "ws.ws_ext_discount_amt,\n"
            "        list_value(\n"
            f"                {outer_args}\n"
            "        ) as outer_prov,\n"
            "        ad.avg_prov as avg_prov"
        )
    else:
        qualified_select = "ws.ws_ext_discount_amt"

    return f"""
with avg_discount as (
    select
        {threshold_key_select}1.3 * avg(
            ws_avg.ws_ext_discount_amt
        ) as discount_threshold
        {avg_prov_sql}
    from {avg_from_sql}
    {avg_where_sql}
    {group_by_sql}
),
qualified as (
    select
        {qualified_select}
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

    if with_provenance:
        final_select = """
select
    sum(ws_ext_discount_amt) as "Excess Discount Amount",
    array_agg(
        list_value(
            list_value(outer_prov),
            avg_prov
        )
    ) as prov
from qualified
order by
    sum(ws_ext_discount_amt)
limit 100
"""
    else:
        final_select = """
select
    sum(ws_ext_discount_amt) as "Excess Discount Amount"
from qualified
order by
    sum(ws_ext_discount_amt)
limit 100
"""

    return ctes + final_select


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

    # Both S1 parameters remain represented in every retained mutation.
    require(
        "i.i_manufact_id =" in lower,
        f"{subtype}: manufacturer predicate is missing.",
    )
    require(
        "d_avg.d_date between" in lower,
        f"{subtype}: sampled date is missing from threshold computation.",
    )

    require(
        "i.i_item_sk = ws.ws_item_sk" in lower,
        f"{subtype}: outer item join is missing.",
    )
    require(
        "d.d_date_sk = ws.ws_sold_date_sk" in lower,
        f"{subtype}: outer date join is missing.",
    )
    require(
        "d_avg.d_date_sk = ws_avg.ws_sold_date_sk" in lower,
        f"{subtype}: threshold-side date join is missing.",
    )

    if subtype == "q92_s2_join_add_web_page":
        require(
            "join web_page wp" in lower,
            "join_add_web_page: relation is missing.",
        )
        require(
            "ws.ws_web_page_sk = wp.wp_web_page_sk" in lower,
            "join_add_web_page: join predicate is missing.",
        )
        require(
            has_alias_prov(lower, "wp"),
            "join_add_web_page: provenance is missing.",
        )

    elif subtype == "q92_s2_join_add_customer":
        require(
            "join customer c" in lower,
            "join_add_customer: relation is missing.",
        )
        require(
            "ws.ws_bill_customer_sk = c.c_customer_sk" in lower,
            "join_add_customer: join predicate is missing.",
        )
        require(
            has_alias_prov(lower, "c"),
            "join_add_customer: provenance is missing.",
        )

    elif subtype == "q92_s2_where_remove_outer_date":
        require(
            "d.d_date between" not in lower,
            "where_remove_outer_date: outer date predicate remains.",
        )

    elif subtype == "q92_s2_where_remove_discount_threshold":
        require(
            "ws.ws_ext_discount_amt > ad.discount_threshold"
            not in lower,
            "where_remove_discount_threshold: threshold predicate remains.",
        )

    elif subtype == "q92_s2_agg_group_by_manufacturer":
        require(
            "join item i_avg" in lower,
            "agg_group_by_manufacturer: threshold-side item is missing.",
        )
        require(
            "i_avg.i_manufact_id as threshold_key" in lower,
            "agg_group_by_manufacturer: threshold key is incorrect.",
        )
        require(
            "group by i_avg.i_manufact_id" in lower,
            "agg_group_by_manufacturer: GROUP BY is incorrect.",
        )
        require(
            "ad.threshold_key = i.i_manufact_id" in lower,
            "agg_group_by_manufacturer: threshold join is incorrect.",
        )
        require(
            has_alias_prov(lower, "i_avg"),
            "agg_group_by_manufacturer: item provenance is missing.",
        )

    if subtype != "q92_s2_where_remove_outer_date":
        require(
            "d.d_date between" in lower,
            f"{subtype}: outer date predicate is missing.",
        )

    if subtype != "q92_s2_where_remove_discount_threshold":
        require(
            "ws.ws_ext_discount_amt > ad.discount_threshold"
            in lower,
            f"{subtype}: threshold predicate is missing.",
        )


# ============================================================
# EXECUTION AND FACTORISED-PROVENANCE VALIDATION
# ============================================================

def contributor_count(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    row: pd.Series,
) -> int:
    sql = render_sql(
        build_contributor_count_sql(subtype),
        row,
    )

    result = con.execute(sql).fetchone()

    if result is None:
        raise RuntimeError(
            f"{subtype}: contributor count returned no row."
        )

    return int(result[0])


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

    result = con.execute(
        f"""
select count(*)
from (
{sql}
) q92_result_count
"""
    ).fetchone()

    if result is None:
        raise RuntimeError(
            f"{subtype}: result count returned no row."
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


def parse_factor(
    factor: Any,
) -> tuple[list[Any], list[list[Any]]]:
    factor = as_python(factor)

    if not isinstance(factor, list) or len(factor) != 2:
        raise RuntimeError(
            f"Unexpected Q92 factor shape: {factor!r}"
        )

    outer_wrapper = as_python(factor[0])
    avg_collection = as_python(factor[1])

    if (
        isinstance(outer_wrapper, list)
        and len(outer_wrapper) == 1
        and isinstance(as_python(outer_wrapper[0]), list)
    ):
        outer_tokens = flatten_scalar_tokens(
            outer_wrapper[0]
        )
    else:
        outer_tokens = flatten_scalar_tokens(
            outer_wrapper
        )

    if avg_collection is None:
        avg_witnesses: list[list[Any]] = []
    else:
        if not isinstance(avg_collection, list):
            avg_collection = [avg_collection]

        avg_witnesses = [
            flatten_scalar_tokens(witness)
            for witness in avg_collection
            if witness is not None
        ]

    return outer_tokens, avg_witnesses


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
    expected_avg = int(
        info["expected_avg_witness_size"]
    )
    expected_expanded = int(
        info["expected_expanded_witness_size"]
    )

    checked = 0

    for factor in prov_cell:
        outer_tokens, avg_witnesses = parse_factor(factor)

        if len(outer_tokens) != expected_outer:
            raise RuntimeError(
                f"{subtype}: expected outer witness size "
                f"{expected_outer}, observed {len(outer_tokens)}."
            )

        if not avg_witnesses:
            raise RuntimeError(
                f"{subtype}: factor has no average witnesses."
            )

        for avg_witness in avg_witnesses:
            if len(avg_witness) != expected_avg:
                raise RuntimeError(
                    f"{subtype}: expected average witness size "
                    f"{expected_avg}, observed {len(avg_witness)}."
                )

            expanded_size = (
                len(outer_tokens) + len(avg_witness)
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
        f"avg={expected_avg} "
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
                    f"{subtype}: source {row['file']} produced "
                    "no qualified tuples during pre-check."
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
    short_name = subtype.replace("q92_s2_", "")

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
                "manufact": int(row["manufact"]),
                "wsdate": str(row["wsdate"]),
                "date_window_days": int(
                    row["date_window_days"]
                ),
                # Q92 metadata uses n_rows for qualified outer tuples,
                # not for the final one-row aggregate.
                "n_rows": int(n_contributors),
                "result_n_rows": int(n_result_rows),
                "normal_n_rows": int(row["n_rows"]),
                "provenance_semantics": (
                    "factorized_avg_predicate"
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
    p = argparse.ArgumentParser(description="Generate Q92 S2 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q92.")
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
    for name in ['q92_s2_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, EXTERNAL_BASE_DIR, N_S2_PER_TYPE, OUT_S2, Q92_NORMAL_DIR, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_S2_PER_TYPE = args.n_per_type
    Q92_NORMAL_DIR = args.normal_dir
    OUT_S2 = args.output_dir / "q92_s2_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print("=" * 96)
    print("[Q92 FINAL S2 GENERATOR]")
    print("=" * 96)
    print(f"DB_PATH={DB_PATH}")
    print(f"DB_SCHEMA={DB_SCHEMA}")
    print(f"Q92_NORMAL_DIR={Q92_NORMAL_DIR}")
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

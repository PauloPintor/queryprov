#!/usr/bin/env python3
from __future__ import annotations

import argparse

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

N_DEMOGRAPHIC_PROFILES = 3
N_STATES = 9

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

Q85_NORMAL_DIR = None  # Set by command-line arguments.
OUT_S2 = None  # Set by command-line arguments.

N_PROV_SHAPE_CHECKS_PER_SUBTYPE = 3
MAX_WITNESSES_CHECKED_PER_QUERY = 500

S2_SUBTYPES = [
    "q85_s2_join_add_item",
    "q85_s2_join_add_customer",
    "q85_s2_join_remove_web_page",
    "q85_s2_where_remove_year",
    "q85_s2_where_remove_sales_price",
    "q85_s2_where_remove_net_profit",
]

S2_INFO: Dict[str, dict] = {
    "q85_s2_join_add_item": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_witness_size": 9,
        "description": (
            "Adds item through ws_item_sk = i_item_sk and appends item "
            "provenance to every witness."
        ),
    },
    "q85_s2_join_add_customer": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_witness_size": 9,
        "description": (
            "Adds the refunded customer through wr_refunded_customer_sk = "
            "c_customer_sk and appends customer provenance to every witness."
        ),
    },
    "q85_s2_join_remove_web_page": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join_removal",
        "logical_unit": "relation and join removal",
        "expected_witness_size": 7,
        "description": (
            "Removes web_page, its join predicate, and web_page provenance "
            "while preserving the remaining Q85 structure."
        ),
    },
    "q85_s2_where_remove_year": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "expected_witness_size": 8,
        "description": (
            "Removes the fixed d_year predicate while preserving the sampled "
            "demographic profiles and states."
        ),
    },
    "q85_s2_where_remove_sales_price": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "expected_witness_size": 8,
        "description": (
            "Removes all three ws_sales_price intervals while preserving the "
            "three normally sampled demographic profiles."
        ),
    },
    "q85_s2_where_remove_net_profit": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "expected_witness_size": 8,
        "description": (
            "Removes all three ws_net_profit intervals while preserving the "
            "nine normally sampled states."
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

    metadata_path = Q85_NORMAL_DIR / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Normal Q85 metadata does not exist: {metadata_path}"
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
    metadata_path = Q85_NORMAL_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)

    required = ["file", "year"]

    for index in range(1, N_DEMOGRAPHIC_PROFILES + 1):
        required.extend([f"ms{index}", f"es{index}"])

    for index in range(1, N_STATES + 1):
        required.append(f"s{index}")

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Normal Q85 metadata is missing columns: {missing}"
        )

    df = df.dropna(subset=required).copy()

    if df.empty:
        raise ValueError("Normal Q85 metadata contains no usable rows.")

    df["file"] = df["file"].astype(str)
    df["year"] = pd.to_numeric(
        df["year"],
        errors="raise",
    ).astype(int)

    for index in range(1, N_DEMOGRAPHIC_PROFILES + 1):
        df[f"ms{index}"] = df[f"ms{index}"].astype(str)
        df[f"es{index}"] = df[f"es{index}"].astype(str)

    for index in range(1, N_STATES + 1):
        df[f"s{index}"] = df[f"s{index}"].astype(str)

    if "n_rows" in df.columns:
        df["n_rows"] = pd.to_numeric(
            df["n_rows"],
            errors="coerce",
        )

    df = df.drop_duplicates(subset=["file"]).reset_index(drop=True)

    return df


def select_common_normal_sources() -> pd.DataFrame:
    """
    Select one deterministic, unconditioned normal sample shared by all
    structural subtypes.
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
    params = {
        "year": int(row["year"]),
    }

    for index in range(1, N_DEMOGRAPHIC_PROFILES + 1):
        params[f"ms{index}"] = sql_escape(
            str(row[f"ms{index}"])
        )
        params[f"es{index}"] = sql_escape(
            str(row[f"es{index}"])
        )

    for index in range(1, N_STATES + 1):
        params[f"s{index}"] = sql_escape(
            str(row[f"s{index}"])
        )

    return params


def demographic_predicate_sql(
    remove_sales_price: bool,
) -> str:
    price_ranges = {
        1: ("100.00", "150.00"),
        2: ("50.00", "100.00"),
        3: ("150.00", "200.00"),
    }

    branches: list[str] = []

    for index in range(1, N_DEMOGRAPHIC_PROFILES + 1):
        conditions = [
            f"cd1.cd_marital_status = '{{ms{index}}}'",
            "cd1.cd_marital_status = cd2.cd_marital_status",
            f"cd1.cd_education_status = '{{es{index}}}'",
            "cd1.cd_education_status = cd2.cd_education_status",
        ]

        if not remove_sales_price:
            lo, hi = price_ranges[index]
            conditions.append(
                f"ws.ws_sales_price between {lo} and {hi}"
            )

        branches.append(
            "(\n"
            "          "
            + "\n      and ".join(conditions)
            + "\n      )"
        )

    return "(\n      " + "\n   or ".join(branches) + "\n  )"


def state_predicate_sql(
    remove_net_profit: bool,
) -> str:
    groups = {
        1: (1, 2, 3, 100, 200),
        2: (4, 5, 6, 150, 300),
        3: (7, 8, 9, 50, 250),
    }

    branches: list[str] = []

    for branch_index in range(1, 4):
        first, second, third, profit_lo, profit_hi = (
            groups[branch_index]
        )

        conditions = [
            "ca.ca_country = 'United States'",
            (
                "ca.ca_state in "
                f"('{{s{first}}}', '{{s{second}}}', '{{s{third}}}')"
            ),
        ]

        if not remove_net_profit:
            conditions.append(
                f"ws.ws_net_profit between {profit_lo} and {profit_hi}"
            )

        branches.append(
            "(\n"
            "          "
            + "\n      and ".join(conditions)
            + "\n      )"
        )

    return "(\n      " + "\n   or ".join(branches) + "\n  )"


def subtype_spec(subtype: str) -> dict:
    add_item = subtype == "q85_s2_join_add_item"
    add_customer = subtype == "q85_s2_join_add_customer"
    remove_web_page = subtype == "q85_s2_join_remove_web_page"
    remove_year = subtype == "q85_s2_where_remove_year"
    remove_sales_price = (
        subtype == "q85_s2_where_remove_sales_price"
    )
    remove_net_profit = (
        subtype == "q85_s2_where_remove_net_profit"
    )

    relations = [
        "web_sales ws",
        "web_returns wr",
    ]

    provenance = [
        "ws.prov",
        "wr.prov",
    ]

    predicates = [
        "ws.ws_item_sk = wr.wr_item_sk",
        "ws.ws_order_number = wr.wr_order_number",
    ]

    if not remove_web_page:
        relations.append("web_page wp")
        provenance.append("wp.prov")
        predicates.append(
            "ws.ws_web_page_sk = wp.wp_web_page_sk"
        )

    relations.extend(
        [
            "customer_demographics cd1",
            "customer_demographics cd2",
            "customer_address ca",
            "date_dim d",
            "reason r",
        ]
    )

    provenance.extend(
        [
            "cd1.prov",
            "cd2.prov",
            "ca.prov",
            "d.prov",
            "r.prov",
        ]
    )

    predicates.append(
        "ws.ws_sold_date_sk = d.d_date_sk"
    )

    if not remove_year:
        predicates.append("d.d_year = {year}")

    predicates.extend(
        [
            "cd1.cd_demo_sk = wr.wr_refunded_cdemo_sk",
            "cd2.cd_demo_sk = wr.wr_returning_cdemo_sk",
            "ca.ca_address_sk = wr.wr_refunded_addr_sk",
            "r.r_reason_sk = wr.wr_reason_sk",
        ]
    )

    if add_item:
        relations.append("item i")
        provenance.append("i.prov")
        predicates.append(
            "ws.ws_item_sk = i.i_item_sk"
        )

    if add_customer:
        relations.append("customer c")
        provenance.append("c.prov")
        predicates.append(
            "wr.wr_refunded_customer_sk = c.c_customer_sk"
        )

    predicates.append(
        demographic_predicate_sql(
            remove_sales_price=remove_sales_price,
        )
    )
    predicates.append(
        state_predicate_sql(
            remove_net_profit=remove_net_profit,
        )
    )

    return {
        "relations": relations,
        "provenance": provenance,
        "predicates": predicates,
    }


def build_query_sql(
    subtype: str,
    with_provenance: bool,
    apply_limit: bool,
) -> str:
    spec = subtype_spec(subtype)

    relations_sql = ",\n     ".join(spec["relations"])
    predicates_sql = "\n  and ".join(spec["predicates"])

    select_expressions = [
        "substr(r.r_reason_desc, 1, 20) as reason_desc",
        "avg(ws.ws_quantity) as avg_ws_quantity",
        "avg(wr.wr_refunded_cash) as avg_wr_refunded_cash",
        "avg(wr.wr_fee) as avg_wr_fee",
    ]

    if with_provenance:
        witness_args = ",\n                ".join(
            spec["provenance"]
        )

        select_expressions.append(
            "array_agg(\n"
            "            list_value(\n"
            f"                {witness_args}\n"
            "            )\n"
            "        ) as prov"
        )

    select_sql = ",\n    ".join(select_expressions)

    limit_sql = "\nlimit 100" if apply_limit else ""

    return f"""
select
    {select_sql}
from {relations_sql}
where {predicates_sql}
group by
    r.r_reason_desc
order by
    substr(r.r_reason_desc, 1, 20),
    avg(ws.ws_quantity),
    avg(wr.wr_refunded_cash),
    avg(wr.wr_fee)
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
        lower.count("cd1.cd_marital_status = '")
        == N_DEMOGRAPHIC_PROFILES,
        f"{subtype}: expected {N_DEMOGRAPHIC_PROFILES} sampled "
        "marital-status predicates.",
    )
    require(
        lower.count("cd1.cd_education_status = '")
        == N_DEMOGRAPHIC_PROFILES,
        f"{subtype}: expected {N_DEMOGRAPHIC_PROFILES} sampled "
        "education predicates.",
    )
    require(
        lower.count("ca.ca_state in") == 3,
        f"{subtype}: expected three state groups.",
    )

    require(
        "ws.ws_item_sk = wr.wr_item_sk" in lower,
        f"{subtype}: sales-return item join is missing.",
    )
    require(
        "ws.ws_order_number = wr.wr_order_number" in lower,
        f"{subtype}: sales-return order join is missing.",
    )
    require(
        "r.r_reason_sk = wr.wr_reason_sk" in lower,
        f"{subtype}: reason join is missing.",
    )

    if subtype == "q85_s2_join_add_item":
        require(
            "item i" in lower,
            "join_add_item: item relation is missing.",
        )
        require(
            "ws.ws_item_sk = i.i_item_sk" in lower,
            "join_add_item: item join predicate is missing.",
        )
        require(
            re.search(
                r"(?<![a-z0-9_])i\.prov(?![a-z0-9_])",
                lower,
            )
            is not None,
            "join_add_item: item provenance is missing.",
        )

    elif subtype == "q85_s2_join_add_customer":
        require(
            "customer c" in lower,
            "join_add_customer: customer relation is missing.",
        )
        require(
            "wr.wr_refunded_customer_sk = c.c_customer_sk" in lower,
            "join_add_customer: customer join predicate is missing.",
        )
        require(
            re.search(
                r"(?<![a-z0-9_])c\.prov(?![a-z0-9_])",
                lower,
            )
            is not None,
            "join_add_customer: customer provenance is missing.",
        )

    elif subtype == "q85_s2_join_remove_web_page":
        require(
            "web_page wp" not in lower,
            "join_remove_web_page: web_page relation remains.",
        )
        require(
            "ws.ws_web_page_sk = wp.wp_web_page_sk" not in lower,
            "join_remove_web_page: web_page join remains.",
        )
        require(
            re.search(
                r"(?<![a-z0-9_])wp\.prov(?![a-z0-9_])",
                lower,
            )
            is None,
            "join_remove_web_page: web_page provenance remains.",
        )

    elif subtype == "q85_s2_where_remove_year":
        require(
            "d.d_year =" not in lower,
            "where_remove_year: year predicate remains.",
        )

    elif subtype == "q85_s2_where_remove_sales_price":
        require(
            "ws.ws_sales_price between" not in lower,
            "where_remove_sales_price: a sales-price interval remains.",
        )

    elif subtype == "q85_s2_where_remove_net_profit":
        require(
            "ws.ws_net_profit between" not in lower,
            "where_remove_net_profit: a net-profit interval remains.",
        )

    if subtype != "q85_s2_join_remove_web_page":
        require(
            "web_page wp" in lower,
            f"{subtype}: web_page relation is missing.",
        )
        require(
            "ws.ws_web_page_sk = wp.wp_web_page_sk" in lower,
            f"{subtype}: web_page join is missing.",
        )

    if subtype != "q85_s2_where_remove_year":
        require(
            "d.d_year =" in lower,
            f"{subtype}: year predicate is missing.",
        )

    if subtype != "q85_s2_where_remove_sales_price":
        require(
            lower.count("ws.ws_sales_price between") == 3,
            f"{subtype}: expected three sales-price intervals.",
        )

    if subtype != "q85_s2_where_remove_net_profit":
        require(
            lower.count("ws.ws_net_profit between") == 3,
            f"{subtype}: expected three net-profit intervals.",
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
) q85_s2_count
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

        witnesses.append(
            flatten_scalar_tokens(witness)
        )

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
# METADATA AND GENERATION
# ============================================================

def subtype_filename(
    subtype: str,
    index: int,
) -> str:
    short_name = subtype.replace("q85_s2_", "")

    return f"s2_{short_name}_{index:05d}.sql"


def metadata_parameter_values(
    row: pd.Series,
) -> dict:
    output = {
        "year": int(row["year"]),
    }

    for index in range(1, N_DEMOGRAPHIC_PROFILES + 1):
        output[f"ms{index}"] = str(row[f"ms{index}"])
        output[f"es{index}"] = str(row[f"es{index}"])

    for index in range(1, N_STATES + 1):
        output[f"s{index}"] = str(row[f"s{index}"])

    if "n_rows" in row.index and not pd.isna(row["n_rows"]):
        # In the original Q85 generator, n_rows counts matching joined tuples
        # before GROUP BY. It is therefore retained only as source metadata.
        output["normal_n_join_rows"] = int(row["n_rows"])

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

        # These checks never trigger resampling. A failure aborts the run.
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
    p = argparse.ArgumentParser(description="Generate Q85 S2 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q85.")
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
    for name in ['q85_s2_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, EXTERNAL_BASE_DIR, N_S2_PER_TYPE, OUT_S2, Q85_NORMAL_DIR, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_S2_PER_TYPE = args.n_per_type
    OUT_S2 = args.output_dir / "q85_s2_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir
    Q85_NORMAL_DIR = args.normal_dir


def main() -> None:
    configure_cli()
    print("=" * 92)
    print("[Q85 FINAL S2 GENERATOR]")
    print("=" * 92)
    print(f"DB_PATH={DB_PATH}")
    print(f"DB_SCHEMA={DB_SCHEMA}")
    print(f"Q85_NORMAL_DIR={Q85_NORMAL_DIR}")
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

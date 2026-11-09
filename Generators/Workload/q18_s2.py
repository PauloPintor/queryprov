#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DB_PATH = None  # Set by command-line arguments.
DB_SCHEMA = "tpcds"

SEED = 42

# Q18 S2 tem 5 tipos. 350 por tipo dá 1750 S2.
# Na escala maior, 625 anomalias / 5 tipos = 125 por tipo.
N_ANOMALY_PER_TYPE = 350

EXTERNAL_Q5_DIR = None  # Set by command-line arguments.
NORMAL_METADATA = None  # Set by command-line arguments.


# Ajusta se quiseres outro sítio. O script só escreve aqui.
OUT_ANOM = None  # Set by command-line arguments.

S2_SUBTYPES = [
    "q18_s2_join_add_table",
    "q18_s2_where_remove_condition",
    "q18_s2_agg_no_group_by",
    "q18_s2_agg_group_by_state",
    "q18_s2_limit_1",
]


# ============================================================
# SQL NORMAL Q18
# ============================================================

SQL_Q18_NORMAL = """
select
       i_item_id,
       ca_country,
       ca_state,
       ca_county,
       avg(cast(cs_quantity as decimal(12,2))) as agg1,
       avg(cast(cs_list_price as decimal(12,2))) as agg2,
       avg(cast(cs_coupon_amt as decimal(12,2))) as agg3,
       avg(cast(cs_sales_price as decimal(12,2))) as agg4,
       avg(cast(cs_net_profit as decimal(12,2))) as agg5,
       avg(cast(c_birth_year as decimal(12,2))) as agg6,
       avg(cast(cd1.cd_dep_count as decimal(12,2))) as agg7,
       array_agg(
           list_value(
               catalog_sales.prov,
               cd1.prov,
               cd2.prov,
               customer.prov,
               customer_address.prov,
               date_dim.prov,
               item.prov
           )
       ) as prov
from catalog_sales,
     customer_demographics cd1,
     customer_demographics cd2,
     customer,
     customer_address,
     date_dim,
     item
where cs_sold_date_sk = d_date_sk
  and cs_item_sk = i_item_sk
  and cs_bill_cdemo_sk = cd1.cd_demo_sk
  and cs_bill_customer_sk = c_customer_sk

  and cd1.cd_gender = '{gen}'
  and cd1.cd_education_status = '{es}'

  and c_current_cdemo_sk = cd2.cd_demo_sk
  and c_current_addr_sk = ca_address_sk

  and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})

  and d_year = {year}

  and ca_state in ('{s1}', '{s2}', '{s3}', '{s4}', '{s5}', '{s6}', '{s7}')

group by rollup (
    i_item_id,
    ca_country,
    ca_state,
    ca_county
)

order by
    ca_country,
    ca_state,
    ca_county,
    i_item_id
"""


COUNT_Q18_NORMAL = """
select count(*) from (
    select
           i_item_id,
           ca_country,
           ca_state,
           ca_county
    from catalog_sales,
         customer_demographics cd1,
         customer_demographics cd2,
         customer,
         customer_address,
         date_dim,
         item
    where cs_sold_date_sk = d_date_sk
      and cs_item_sk = i_item_sk
      and cs_bill_cdemo_sk = cd1.cd_demo_sk
      and cs_bill_customer_sk = c_customer_sk

      and cd1.cd_gender = '{gen}'
      and cd1.cd_education_status = '{es}'

      and c_current_cdemo_sk = cd2.cd_demo_sk
      and c_current_addr_sk = ca_address_sk

      and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})

      and d_year = {year}

      and ca_state in ('{s1}', '{s2}', '{s3}', '{s4}', '{s5}', '{s6}', '{s7}')

    group by rollup (
        i_item_id,
        ca_country,
        ca_state,
        ca_county
    )
) t
"""


# ============================================================
# S2 1: JOIN ADD TABLE
# Adds catalog_page through cs_catalog_page_sk = cp_catalog_page_sk.
# ============================================================

SQL_Q18_S2_JOIN_ADD_TABLE = """
select
       i_item_id,
       ca_country,
       ca_state,
       ca_county,
       avg(cast(cs_quantity as decimal(12,2))) as agg1,
       avg(cast(cs_list_price as decimal(12,2))) as agg2,
       avg(cast(cs_coupon_amt as decimal(12,2))) as agg3,
       avg(cast(cs_sales_price as decimal(12,2))) as agg4,
       avg(cast(cs_net_profit as decimal(12,2))) as agg5,
       avg(cast(c_birth_year as decimal(12,2))) as agg6,
       avg(cast(cd1.cd_dep_count as decimal(12,2))) as agg7,
       array_agg(
           list_value(
               catalog_sales.prov,
               catalog_page.prov,
               cd1.prov,
               cd2.prov,
               customer.prov,
               customer_address.prov,
               date_dim.prov,
               item.prov
           )
       ) as prov
from catalog_sales,
     catalog_page,
     customer_demographics cd1,
     customer_demographics cd2,
     customer,
     customer_address,
     date_dim,
     item
where cs_sold_date_sk = d_date_sk
  and cs_item_sk = i_item_sk
  and cs_catalog_page_sk = cp_catalog_page_sk
  and cs_bill_cdemo_sk = cd1.cd_demo_sk
  and cs_bill_customer_sk = c_customer_sk

  and cd1.cd_gender = '{gen}'
  and cd1.cd_education_status = '{es}'

  and c_current_cdemo_sk = cd2.cd_demo_sk
  and c_current_addr_sk = ca_address_sk

  and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})

  and d_year = {year}

  and ca_state in ('{s1}', '{s2}', '{s3}', '{s4}', '{s5}', '{s6}', '{s7}')

group by rollup (
    i_item_id,
    ca_country,
    ca_state,
    ca_county
)

order by
    ca_country,
    ca_state,
    ca_county,
    i_item_id
"""


COUNT_Q18_S2_JOIN_ADD_TABLE = """
select count(*) from (
    select
           i_item_id,
           ca_country,
           ca_state,
           ca_county
    from catalog_sales,
         catalog_page,
         customer_demographics cd1,
         customer_demographics cd2,
         customer,
         customer_address,
         date_dim,
         item
    where cs_sold_date_sk = d_date_sk
      and cs_item_sk = i_item_sk
      and cs_catalog_page_sk = cp_catalog_page_sk
      and cs_bill_cdemo_sk = cd1.cd_demo_sk
      and cs_bill_customer_sk = c_customer_sk

      and cd1.cd_gender = '{gen}'
      and cd1.cd_education_status = '{es}'

      and c_current_cdemo_sk = cd2.cd_demo_sk
      and c_current_addr_sk = ca_address_sk

      and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})

      and d_year = {year}

      and ca_state in ('{s1}', '{s2}', '{s3}', '{s4}', '{s5}', '{s6}', '{s7}')

    group by rollup (
        i_item_id,
        ca_country,
        ca_state,
        ca_county
    )
) t
"""


# ============================================================
# S2 2: WHERE REMOVE CONDITION
# Removes only c_birth_month in (...).
# Preserves gen/es/year/states.
# ============================================================

SQL_Q18_S2_WHERE_REMOVE_CONDITION = SQL_Q18_NORMAL.replace(
    """
  and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})
""",
    ""
)

COUNT_Q18_S2_WHERE_REMOVE_CONDITION = COUNT_Q18_NORMAL.replace(
    """
      and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})
""",
    ""
)


# ============================================================
# S2 3: AGG NO GROUP BY
# Global aggregation, no rollup keys.
# ============================================================

SQL_Q18_S2_AGG_NO_GROUP_BY = """
select
       avg(cast(cs_quantity as decimal(12,2))) as agg1,
       avg(cast(cs_list_price as decimal(12,2))) as agg2,
       avg(cast(cs_coupon_amt as decimal(12,2))) as agg3,
       avg(cast(cs_sales_price as decimal(12,2))) as agg4,
       avg(cast(cs_net_profit as decimal(12,2))) as agg5,
       avg(cast(c_birth_year as decimal(12,2))) as agg6,
       avg(cast(cd1.cd_dep_count as decimal(12,2))) as agg7,
       array_agg(
           list_value(
               catalog_sales.prov,
               cd1.prov,
               cd2.prov,
               customer.prov,
               customer_address.prov,
               date_dim.prov,
               item.prov
           )
       ) as prov
from catalog_sales,
     customer_demographics cd1,
     customer_demographics cd2,
     customer,
     customer_address,
     date_dim,
     item
where cs_sold_date_sk = d_date_sk
  and cs_item_sk = i_item_sk
  and cs_bill_cdemo_sk = cd1.cd_demo_sk
  and cs_bill_customer_sk = c_customer_sk

  and cd1.cd_gender = '{gen}'
  and cd1.cd_education_status = '{es}'

  and c_current_cdemo_sk = cd2.cd_demo_sk
  and c_current_addr_sk = ca_address_sk

  and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})

  and d_year = {year}

  and ca_state in ('{s1}', '{s2}', '{s3}', '{s4}', '{s5}', '{s6}', '{s7}')
"""


COUNT_Q18_S2_AGG_NO_GROUP_BY = """
select count(*) as n_rows
from (
""" + SQL_Q18_S2_AGG_NO_GROUP_BY.strip().rstrip(";") + """
) t
where agg1 is not null
"""


# ============================================================
# S2 4: AGG GROUP BY STATE
# Coarser geographical aggregation.
# ============================================================

SQL_Q18_S2_AGG_GROUP_BY_STATE = """
select
       ca_country,
       ca_state,
       avg(cast(cs_quantity as decimal(12,2))) as agg1,
       avg(cast(cs_list_price as decimal(12,2))) as agg2,
       avg(cast(cs_coupon_amt as decimal(12,2))) as agg3,
       avg(cast(cs_sales_price as decimal(12,2))) as agg4,
       avg(cast(cs_net_profit as decimal(12,2))) as agg5,
       avg(cast(c_birth_year as decimal(12,2))) as agg6,
       avg(cast(cd1.cd_dep_count as decimal(12,2))) as agg7,
       array_agg(
           list_value(
               catalog_sales.prov,
               cd1.prov,
               cd2.prov,
               customer.prov,
               customer_address.prov,
               date_dim.prov,
               item.prov
           )
       ) as prov
from catalog_sales,
     customer_demographics cd1,
     customer_demographics cd2,
     customer,
     customer_address,
     date_dim,
     item
where cs_sold_date_sk = d_date_sk
  and cs_item_sk = i_item_sk
  and cs_bill_cdemo_sk = cd1.cd_demo_sk
  and cs_bill_customer_sk = c_customer_sk

  and cd1.cd_gender = '{gen}'
  and cd1.cd_education_status = '{es}'

  and c_current_cdemo_sk = cd2.cd_demo_sk
  and c_current_addr_sk = ca_address_sk

  and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})

  and d_year = {year}

  and ca_state in ('{s1}', '{s2}', '{s3}', '{s4}', '{s5}', '{s6}', '{s7}')

group by
    ca_country,
    ca_state

order by
    ca_country,
    ca_state
"""


COUNT_Q18_S2_AGG_GROUP_BY_STATE = """
select count(*) from (
    select
           ca_country,
           ca_state
    from catalog_sales,
         customer_demographics cd1,
         customer_demographics cd2,
         customer,
         customer_address,
         date_dim,
         item
    where cs_sold_date_sk = d_date_sk
      and cs_item_sk = i_item_sk
      and cs_bill_cdemo_sk = cd1.cd_demo_sk
      and cs_bill_customer_sk = c_customer_sk

      and cd1.cd_gender = '{gen}'
      and cd1.cd_education_status = '{es}'

      and c_current_cdemo_sk = cd2.cd_demo_sk
      and c_current_addr_sk = ca_address_sk

      and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})

      and d_year = {year}

      and ca_state in ('{s1}', '{s2}', '{s3}', '{s4}', '{s5}', '{s6}', '{s7}')

    group by
        ca_country,
        ca_state
) t
"""


# ============================================================
# S2 5: LIMIT 1
# Q18 original has no LIMIT. Adding LIMIT 1 is restrictive.
# ============================================================

SQL_Q18_S2_LIMIT_1 = SQL_Q18_NORMAL.rstrip() + "\nlimit 1\n"

COUNT_Q18_S2_LIMIT_1 = """
select count(*) from (
""" + SQL_Q18_S2_LIMIT_1.strip().rstrip(";") + """
) t
"""


SQL_BY_SUBTYPE = {
    "q18_s2_join_add_table": SQL_Q18_S2_JOIN_ADD_TABLE,
    "q18_s2_where_remove_condition": SQL_Q18_S2_WHERE_REMOVE_CONDITION,
    "q18_s2_agg_no_group_by": SQL_Q18_S2_AGG_NO_GROUP_BY,
    "q18_s2_agg_group_by_state": SQL_Q18_S2_AGG_GROUP_BY_STATE,
    "q18_s2_limit_1": SQL_Q18_S2_LIMIT_1,
}


COUNT_BY_SUBTYPE = {
    "q18_s2_join_add_table": COUNT_Q18_S2_JOIN_ADD_TABLE,
    "q18_s2_where_remove_condition": COUNT_Q18_S2_WHERE_REMOVE_CONDITION,
    "q18_s2_agg_no_group_by": COUNT_Q18_S2_AGG_NO_GROUP_BY,
    "q18_s2_agg_group_by_state": COUNT_Q18_S2_AGG_GROUP_BY_STATE,
    "q18_s2_limit_1": COUNT_Q18_S2_LIMIT_1,
}


# ============================================================
# HELPERS
# ============================================================

def sql_escape(s: str) -> str:
    return str(s).replace("'", "''")


def prepare_sql_params(params: dict) -> dict:
    p = dict(params)

    for key in [
        "gen",
        "es",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
    ]:
        p[key] = sql_escape(p[key])

    return p


def render(template: str, params: dict) -> str:
    return template.format(**prepare_sql_params(params))


def load_normal_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Normal metadata not found: {path}")

    df = pd.read_csv(path)

    required = {
        "gen",
        "es",
        "year",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Normal metadata must contain {sorted(required)}. Missing: {sorted(missing)}"
        )

    df = df.copy()

    for c in ["gen", "es", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]:
        df[c] = df[c].astype(str)

    for c in ["year", "m1", "m2", "m3", "m4", "m5", "m6"]:
        df[c] = df[c].astype(int)

    if len(df) == 0:
        raise ValueError("Normal metadata is empty.")

    return df


def write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def remove_old_s2_files(out_dir: Path) -> None:
    if not out_dir.exists():
        return

    for subtype in S2_SUBTYPES:
        for p in out_dir.glob(f"{subtype}_*.sql"):
            p.unlink()

    for name in [
        "metadata_s2.csv",
        "q18_s2_metadata.csv",
        "q18_s2_summary.csv",
    ]:
        p = out_dir / name
        if p.exists():
            p.unlink()


def sample_normal_params(
    normal_df: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    row = normal_df.iloc[int(rng.integers(0, len(normal_df)))]

    params = {
        "year": int(row["year"]),
        "gen": str(row["gen"]),
        "es": str(row["es"]),
        "m1": int(row["m1"]),
        "m2": int(row["m2"]),
        "m3": int(row["m3"]),
        "m4": int(row["m4"]),
        "m5": int(row["m5"]),
        "m6": int(row["m6"]),
        "s1": str(row["s1"]),
        "s2": str(row["s2"]),
        "s3": str(row["s3"]),
        "s4": str(row["s4"]),
        "s5": str(row["s5"]),
        "s6": str(row["s6"]),
        "s7": str(row["s7"]),
    }

    if "file" in normal_df.columns:
        params["source_normal_file"] = str(row["file"])

    return params


def param_signature(subtype: str, params: dict) -> tuple:
    return (
        subtype,
        params["gen"],
        params["es"],
        params["year"],
        params["m1"],
        params["m2"],
        params["m3"],
        params["m4"],
        params["m5"],
        params["m6"],
        tuple(sorted([
            params["s1"],
            params["s2"],
            params["s3"],
            params["s4"],
            params["s5"],
            params["s6"],
            params["s7"],
        ])),
    )


def count_rows(
    con: duckdb.DuckDBPyConnection,
    template: str,
    params: dict,
) -> int:
    return int(con.execute(render(template, params)).fetchone()[0])


# ============================================================
# GENERATION
# ============================================================

def write_s2_queries(
    out_dir: Path,
    subtype: str,
    n: int,
    normal_df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    rng: np.random.Generator,
    allow_duplicates: bool,
) -> list[dict]:

    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    i = 0
    attempts = 0
    seen: set[tuple] = set()

    max_attempts = max(100_000, n * 100)

    sql_template = SQL_BY_SUBTYPE[subtype]
    count_template = COUNT_BY_SUBTYPE[subtype]

    while i < n:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Too many attempts while generating {subtype}: "
                f"{attempts} attempts for {i}/{n} queries. "
                "Try --allow-duplicates or reduce --n-per-type."
            )

        params = sample_normal_params(normal_df, rng)

        sig = param_signature(subtype, params)

        if not allow_duplicates and sig in seen:
            continue

        seen.add(sig)

        try:
            s2_n_rows = count_rows(con, count_template, params)
            normal_n_rows = count_rows(con, COUNT_Q18_NORMAL, params)
        except Exception as e:
            print(f"[ERROR] count failed for {subtype} {params}: {e}")
            continue

        if s2_n_rows == 0:
            continue

        filename = f"{subtype}_{i:05d}.sql"

        (out_dir / filename).write_text(
            render(sql_template, params).strip() + "\n",
            encoding="utf-8",
        )

        row = {
            "file": filename,
            "label": 1,
            "scenario": subtype,
            "year": int(params["year"]),
            "gen": params["gen"],
            "es": params["es"],
            "m1": int(params["m1"]),
            "m2": int(params["m2"]),
            "m3": int(params["m3"]),
            "m4": int(params["m4"]),
            "m5": int(params["m5"]),
            "m6": int(params["m6"]),
            "s1": params["s1"],
            "s2": params["s2"],
            "s3": params["s3"],
            "s4": params["s4"],
            "s5": params["s5"],
            "s6": params["s6"],
            "s7": params["s7"],
            "normal_n_rows": normal_n_rows,
            "s2_n_rows": s2_n_rows,
        }

        if "source_normal_file" in params:
            row["source_normal_file"] = params["source_normal_file"]

        rows.append(row)

        i += 1

        if i % 100 == 0:
            print(f"  [GEN] {subtype}: {i}/{n}")

    print(f"  [OK] {subtype}: {len(rows)} queries, attempts={attempts}")

    return rows


def merge_metadata(
    out_dir: Path,
    s2_rows: list[dict],
) -> None:
    metadata_path = out_dir / "metadata.csv"

    if metadata_path.exists():
        old = pd.read_csv(metadata_path)

        if "scenario" in old.columns:
            old = old[~old["scenario"].astype(str).isin(S2_SUBTYPES)].copy()
        else:
            old = old.copy()

        new = pd.DataFrame(s2_rows)

        cols = list(old.columns)
        for c in new.columns:
            if c not in cols:
                cols.append(c)

        merged = pd.concat(
            [
                old.reindex(columns=cols),
                new.reindex(columns=cols),
            ],
            ignore_index=True,
        )
    else:
        merged = pd.DataFrame(s2_rows)

    merged.to_csv(metadata_path, index=False)
    print(f"[WRITE] {metadata_path} rows={len(merged)}")


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_diagnostics(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("[Q18 S2 GENERATION DIAGNOSTICS]")
    print("=" * 80)

    print(f"\n[TOTAL S2] n={len(df)}")
    print(f"  scenarios={df['scenario'].value_counts().to_dict()}")

    print("\n[PARAMS FROM NORMAL DISTRIBUTION]")
    print(f"  demographic profiles unique={len(set(zip(df['gen'], df['es'])))}")
    print(f"  year unique={sorted(df['year'].unique().astype(int).tolist())}")
    print(f"  months signatures unique={len(set(tuple(r) for r in df[['m1','m2','m3','m4','m5','m6']].to_numpy()))}")

    all_states = (
        set(df["s1"].astype(str))
        | set(df["s2"].astype(str))
        | set(df["s3"].astype(str))
        | set(df["s4"].astype(str))
        | set(df["s5"].astype(str))
        | set(df["s6"].astype(str))
        | set(df["s7"].astype(str))
    )
    print(f"  states unique={len(all_states)}")

    print("\n[ROWS]")
    summary = (
        df.groupby("scenario")
        .agg(
            n=("file", "count"),
            normal_n_rows_min=("normal_n_rows", "min"),
            normal_n_rows_median=("normal_n_rows", "median"),
            normal_n_rows_max=("normal_n_rows", "max"),
            s2_n_rows_min=("s2_n_rows", "min"),
            s2_n_rows_median=("s2_n_rows", "median"),
            s2_n_rows_max=("s2_n_rows", "max"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate Q18 S2 anomalies only. "
            "Read normal metadata and write SQL files and S2 metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--db", "--db-path", dest="db_path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--schema", default=DB_SCHEMA, help="Database search path.")
    p.add_argument("--normal-metadata", type=Path, required=True, help="Metadata CSV of the normal Q18 workload.")

    output = p.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output-dir", type=Path,
        help="Parent directory; output is written to its q18_s2_anomalies subfolder.",
    )
    output.add_argument(
        "--out-anom",
        type=Path,
        help="Legacy alternative: specify the S2 output pool directly.",
    )

    p.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    p.add_argument("--n-per-type", type=int, default=N_ANOMALY_PER_TYPE, help="Number of queries per S2 subtype.")

    p.add_argument(
        "--clean-s2",
        action="store_true",
        help=(
            "Explicitly regenerate S2 outputs in an existing pool. "
            "Remove old generated S2 files; do not delete unrelated SQL files."
        ),
    )

    p.add_argument(
        "--no-merge-metadata",
        action="store_true",
        help=(
            "Do not update metadata.csv in the output pool. "
            "The other three original CSV outputs are still written."
        ),
    )

    p.add_argument("--allow-duplicates", action="store_true", help="Allow repeated parameter signatures; original default is false.")
    args = p.parse_args(argv)
    args.db_path = args.db_path.expanduser().resolve()
    args.normal_metadata = args.normal_metadata.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.output_dir.exists() and not args.output_dir.is_dir():
            p.error(f"Output path is not a directory: {args.output_dir}")
        args.out_anom = args.output_dir / "q18_s2_anomalies"
    else:
        args.out_anom = args.out_anom.expanduser().resolve()
    if not args.db_path.is_file():
        p.error(f"Database file does not exist: {args.db_path}")
    if not args.normal_metadata.is_file():
        p.error(f"Normal metadata file does not exist: {args.normal_metadata}")
    if args.seed < 0:
        p.error("--seed must be non-negative.")
    if args.n_per_type <= 0:
        p.error("--n-per-type must be positive.")
    if not args.schema.strip():
        p.error("--schema must not be empty.")
    if args.normal_metadata.is_relative_to(args.out_anom):
        p.error("Normal input metadata must be outside the S2 output pool.")
    if args.out_anom.exists():
        if not args.out_anom.is_dir():
            p.error(f"Output pool is not a directory: {args.out_anom}")
        if not args.clean_s2 and next(args.out_anom.iterdir(), None) is not None:
            p.error(f"Output pool is not empty: {args.out_anom}. Use a new output directory or explicitly pass --clean-s2.")
    return args


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("[Q18 S2 QUERY GENERATOR]")
    print("=" * 80)
    print(f"DB_PATH={args.db_path}")
    print(f"DB_SCHEMA={args.schema}")
    print(f"NORMAL_METADATA={args.normal_metadata}")
    print(f"OUT_ANOM={args.out_anom}")
    print(f"SEED={args.seed}")
    print(f"N_ANOMALY_PER_TYPE={args.n_per_type}")
    print(f"ALLOW_DUPLICATES={args.allow_duplicates}")
    print(f"S2_SUBTYPES={S2_SUBTYPES}")
    print("=" * 80)


    normal_df = load_normal_metadata(args.normal_metadata)

    print(f"\n[NORMAL METADATA] rows={len(normal_df)}")
    print(f"  demographic profiles unique={len(set(zip(normal_df['gen'], normal_df['es'])))}")
    print(f"  year unique={sorted(normal_df['year'].unique().astype(int).tolist())}")
    print(f"  months signatures unique={len(set(tuple(r) for r in normal_df[['m1','m2','m3','m4','m5','m6']].to_numpy()))}")

    all_states = (
        set(normal_df["s1"].astype(str))
        | set(normal_df["s2"].astype(str))
        | set(normal_df["s3"].astype(str))
        | set(normal_df["s4"].astype(str))
        | set(normal_df["s5"].astype(str))
        | set(normal_df["s6"].astype(str))
        | set(normal_df["s7"].astype(str))
    )
    print(f"  states unique={len(all_states)}")

    con = duckdb.connect(str(args.db_path), read_only=True)
    con.execute("SET search_path = ?", [args.schema])

    if args.clean_s2:
        print("\n[CLEAN] removing old Q18 S2 files only")
        remove_old_s2_files(args.out_anom)

    rng = np.random.default_rng(args.seed)

    all_rows: list[dict] = []

    for subtype in S2_SUBTYPES:
        print(f"\n[GEN] {subtype}")

        rows = write_s2_queries(
            out_dir=args.out_anom,
            subtype=subtype,
            n=args.n_per_type,
            normal_df=normal_df,
            con=con,
            rng=rng,
            allow_duplicates=args.allow_duplicates,
        )

        all_rows.extend(rows)

    con.close()

    metadata_s2 = args.out_anom / "metadata_s2.csv"
    write_metadata(metadata_s2, all_rows)
    print(f"\n[WRITE] {metadata_s2} rows={len(all_rows)}")

    q18_s2_metadata = args.out_anom / "q18_s2_metadata.csv"
    write_metadata(q18_s2_metadata, all_rows)
    print(f"[WRITE] {q18_s2_metadata} rows={len(all_rows)}")

    summary = (
        pd.DataFrame(all_rows)
        .groupby("scenario")
        .agg(
            n=("file", "count"),
            normal_n_rows_min=("normal_n_rows", "min"),
            normal_n_rows_median=("normal_n_rows", "median"),
            normal_n_rows_max=("normal_n_rows", "max"),
            s2_n_rows_min=("s2_n_rows", "min"),
            s2_n_rows_median=("s2_n_rows", "median"),
            s2_n_rows_max=("s2_n_rows", "max"),
        )
        .reset_index()
    )

    summary_path = args.out_anom / "q18_s2_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[WRITE] {summary_path} rows={len(summary)}")

    if not args.no_merge_metadata:
        merge_metadata(args.out_anom, all_rows)

    print_diagnostics(all_rows)

    print("\n[DONE]")


if __name__ == "__main__":
    main()

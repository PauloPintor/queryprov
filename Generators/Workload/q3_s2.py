#!/usr/bin/env python3
"""
Generate Q3 S2 SQL workloads using parameters from normal metadata.

Example:
    python generate_q3_s2.py --db /path/tpcds_prov.duckdb \
        --normal-metadata Queries/Q3/q3_normal/metadata.csv --output-dir Queries/Q3

Requires duckdb, numpy and pandas. The DuckDB database is opened read-only.
The default produces 600 SQL files for each of the five original subtypes.
SQL templates, sampling, metadata outputs and their original defaults are
preserved.
"""
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

DB_SCHEMA = "tpcds"

SEED = 42

# 600 queries per subtype, 3,000 S2 queries in total.
N_ANOMALY_PER_TYPE = 600

S2_SUBTYPES = [
    "q3_s2_join_add_table",
    "q3_s2_where_remove_condition",
    "q3_s2_agg_no_group_by",
    "q3_s2_agg_group_by_year",
    "q3_s2_limit_1",
]


# ============================================================
# SQL TEMPLATES
# ============================================================

# NORMAL Q3 reference:
#
# - parameters come from q3_normal/metadata.csv
# - S2 keeps the normal parameter distribution
# - S2 changes query structure only


# ------------------------------------------------------------
# S2: add table + valid join
# ------------------------------------------------------------

SQL_Q3_S2_JOIN_ADD_TABLE = """
select
    dt.d_year,
    item.i_brand_id as brand_id,
    item.i_brand as brand,
    sum(ss_ext_sales_price) as sum_agg,
    array_agg(
        list_value(
            dt.prov,
            store_sales.prov,
            item.prov,
            customer.prov
        )
    ) as prov
from date_dim dt,
     store_sales,
     item,
     customer
where dt.d_date_sk = store_sales.ss_sold_date_sk
  and store_sales.ss_item_sk = item.i_item_sk
  and store_sales.ss_customer_sk = customer.c_customer_sk
  and item.i_manufact_id = {manufact}
  and dt.d_moy = {month}
group by
    dt.d_year,
    item.i_brand,
    item.i_brand_id
order by
    dt.d_year,
    sum_agg desc,
    brand_id
limit 100
"""


COUNT_Q3_S2_JOIN_ADD_TABLE = """
select count(*) as n_rows
from (
    select
        dt.d_year,
        item.i_brand,
        item.i_brand_id
    from date_dim dt,
         store_sales,
         item,
         customer
    where dt.d_date_sk = store_sales.ss_sold_date_sk
      and store_sales.ss_item_sk = item.i_item_sk
      and store_sales.ss_customer_sk = customer.c_customer_sk
      and item.i_manufact_id = {manufact}
      and dt.d_moy = {month}
    group by
        dt.d_year,
        item.i_brand,
        item.i_brand_id
) t
"""


# ------------------------------------------------------------
# S2: remove non-join WHERE condition
# Q3 concrete application: remove dt.d_moy = {month}
# ------------------------------------------------------------

SQL_Q3_S2_WHERE_REMOVE_CONDITION = """
select
    dt.d_year,
    item.i_brand_id as brand_id,
    item.i_brand as brand,
    sum(ss_ext_sales_price) as sum_agg,
    array_agg(
        list_value(
            dt.prov,
            store_sales.prov,
            item.prov
        )
    ) as prov
from date_dim dt,
     store_sales,
     item
where dt.d_date_sk = store_sales.ss_sold_date_sk
  and store_sales.ss_item_sk = item.i_item_sk
  and item.i_manufact_id = {manufact}
group by
    dt.d_year,
    item.i_brand,
    item.i_brand_id
order by
    dt.d_year,
    sum_agg desc,
    brand_id
limit 100
"""


COUNT_Q3_S2_WHERE_REMOVE_CONDITION = """
select count(*) as n_rows
from (
    select
        dt.d_year,
        item.i_brand,
        item.i_brand_id
    from date_dim dt,
         store_sales,
         item
    where dt.d_date_sk = store_sales.ss_sold_date_sk
      and store_sales.ss_item_sk = item.i_item_sk
      and item.i_manufact_id = {manufact}
    group by
        dt.d_year,
        item.i_brand,
        item.i_brand_id
) t
"""


# ------------------------------------------------------------
# S2: no GROUP BY
# ------------------------------------------------------------

SQL_Q3_S2_AGG_NO_GROUP_BY = """
select
    sum(ss_ext_sales_price) as sum_agg,
    array_agg(
        list_value(
            dt.prov,
            store_sales.prov,
            item.prov
        )
    ) as prov
from date_dim dt,
     store_sales,
     item
where dt.d_date_sk = store_sales.ss_sold_date_sk
  and store_sales.ss_item_sk = item.i_item_sk
  and item.i_manufact_id = {manufact}
  and dt.d_moy = {month}
limit 100
"""


COUNT_Q3_S2_AGG_NO_GROUP_BY = """
select count(*) as n_rows
from (
    select
        sum(ss_ext_sales_price) as sum_agg
    from date_dim dt,
         store_sales,
         item
    where dt.d_date_sk = store_sales.ss_sold_date_sk
      and store_sales.ss_item_sk = item.i_item_sk
      and item.i_manufact_id = {manufact}
      and dt.d_moy = {month}
) t
where sum_agg is not null
"""


# ------------------------------------------------------------
# S2: coarser GROUP BY, year only
# ------------------------------------------------------------

SQL_Q3_S2_AGG_GROUP_BY_YEAR = """
select
    dt.d_year,
    sum(ss_ext_sales_price) as sum_agg,
    array_agg(
        list_value(
            dt.prov,
            store_sales.prov,
            item.prov
        )
    ) as prov
from date_dim dt,
     store_sales,
     item
where dt.d_date_sk = store_sales.ss_sold_date_sk
  and store_sales.ss_item_sk = item.i_item_sk
  and item.i_manufact_id = {manufact}
  and dt.d_moy = {month}
group by
    dt.d_year
order by
    dt.d_year,
    sum_agg desc
limit 100
"""


COUNT_Q3_S2_AGG_GROUP_BY_YEAR = """
select count(*) as n_rows
from (
    select
        dt.d_year
    from date_dim dt,
         store_sales,
         item
    where dt.d_date_sk = store_sales.ss_sold_date_sk
      and store_sales.ss_item_sk = item.i_item_sk
      and item.i_manufact_id = {manufact}
      and dt.d_moy = {month}
    group by
        dt.d_year
) t
"""


# ------------------------------------------------------------
# S2: LIMIT 1
# ------------------------------------------------------------

SQL_Q3_S2_LIMIT_1 = """
select
    dt.d_year,
    item.i_brand_id as brand_id,
    item.i_brand as brand,
    sum(ss_ext_sales_price) as sum_agg,
    array_agg(
        list_value(
            dt.prov,
            store_sales.prov,
            item.prov
        )
    ) as prov
from date_dim dt,
     store_sales,
     item
where dt.d_date_sk = store_sales.ss_sold_date_sk
  and store_sales.ss_item_sk = item.i_item_sk
  and item.i_manufact_id = {manufact}
  and dt.d_moy = {month}
group by
    dt.d_year,
    item.i_brand,
    item.i_brand_id
order by
    dt.d_year,
    sum_agg desc,
    brand_id
limit 1
"""


COUNT_Q3_S2_LIMIT_1 = """
select count(*) as n_rows
from (
    select
        dt.d_year,
        item.i_brand,
        item.i_brand_id
    from date_dim dt,
         store_sales,
         item
    where dt.d_date_sk = store_sales.ss_sold_date_sk
      and store_sales.ss_item_sk = item.i_item_sk
      and item.i_manufact_id = {manufact}
      and dt.d_moy = {month}
    group by
        dt.d_year,
        item.i_brand,
        item.i_brand_id
    limit 1
) t
"""


SQL_BY_SUBTYPE = {
    "q3_s2_join_add_table": SQL_Q3_S2_JOIN_ADD_TABLE,
    "q3_s2_where_remove_condition": SQL_Q3_S2_WHERE_REMOVE_CONDITION,
    "q3_s2_agg_no_group_by": SQL_Q3_S2_AGG_NO_GROUP_BY,
    "q3_s2_agg_group_by_year": SQL_Q3_S2_AGG_GROUP_BY_YEAR,
    "q3_s2_limit_1": SQL_Q3_S2_LIMIT_1,
}


COUNT_BY_SUBTYPE = {
    "q3_s2_join_add_table": COUNT_Q3_S2_JOIN_ADD_TABLE,
    "q3_s2_where_remove_condition": COUNT_Q3_S2_WHERE_REMOVE_CONDITION,
    "q3_s2_agg_no_group_by": COUNT_Q3_S2_AGG_NO_GROUP_BY,
    "q3_s2_agg_group_by_year": COUNT_Q3_S2_AGG_GROUP_BY_YEAR,
    "q3_s2_limit_1": COUNT_Q3_S2_LIMIT_1,
}


# ============================================================
# NORMAL COUNT FOR DIAGNOSTICS
# ============================================================

COUNT_Q3_NORMAL = """
select count(*) as n_rows
from (
    select
        dt.d_year,
        item.i_brand,
        item.i_brand_id
    from date_dim dt,
         store_sales,
         item
    where dt.d_date_sk = store_sales.ss_sold_date_sk
      and store_sales.ss_item_sk = item.i_item_sk
      and item.i_manufact_id = {manufact}
      and dt.d_moy = {month}
    group by
        dt.d_year,
        item.i_brand,
        item.i_brand_id
) t
"""


# ============================================================
# HELPERS
# ============================================================

def render(template: str, params: dict) -> str:
    return template.format(**params)


def write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)


def load_normal_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Normal metadata not found: {path}")

    df = pd.read_csv(path)

    required = {"manufact", "month"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Normal metadata must contain {sorted(required)}. "
            f"Missing: {sorted(missing)}"
        )

    df = df.copy()
    df["manufact"] = df["manufact"].astype(int)
    df["month"] = df["month"].astype(int)

    if len(df) == 0:
        raise ValueError("Normal metadata is empty.")

    return df


def remove_old_s2_files(out_dir: Path) -> None:
    """
    Remove only old Q3 S2 files generated by this script.
    It does NOT remove S1 files and does NOT delete the q3_anomalies folder.
    """

    if not out_dir.exists():
        return

    for subtype in S2_SUBTYPES:
        for p in out_dir.glob(f"{subtype}_*.sql"):
            p.unlink()

    for name in [
        "metadata_s2.csv",
        "q3_s2_metadata.csv",
        "q3_s2_summary.csv",
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
        "manufact": int(row["manufact"]),
        "month": int(row["month"]),
    }

    if "file" in normal_df.columns:
        params["source_normal_file"] = str(row["file"])

    return params


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
) -> list[dict]:

    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    i = 0
    attempts = 0

    max_attempts = max(100_000, n * 50)

    sql_template = SQL_BY_SUBTYPE[subtype]
    count_template = COUNT_BY_SUBTYPE[subtype]

    while i < n:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Too many attempts while generating {subtype}: "
                f"{attempts} attempts for {i}/{n} queries. "
                "The parameter space may be too small."
            )

        params = sample_normal_params(normal_df, rng)

        try:
            normal_n_rows = count_rows(con, COUNT_Q3_NORMAL, params)
            s2_n_rows = count_rows(con, count_template, params)
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
            "manufact": int(params["manufact"]),
            "month": int(params["month"]),
            "normal_n_rows": normal_n_rows,
            "s2_n_rows": s2_n_rows,
        }

        if "source_normal_file" in params:
            row["source_normal_file"] = params["source_normal_file"]

        rows.append(row)

        i += 1

        if i % 1000 == 0:
            print(f"  [GEN] {subtype}: {i}/{n}")

    print(f"  [OK] {subtype}: {len(rows)} queries, attempts={attempts}")

    return rows


def merge_metadata(
    out_dir: Path,
    s2_rows: list[dict],
) -> None:
    """
    Keeps old S1 metadata if it exists, removes old S2 rows, then appends new S2 rows.

    This way q3_anomalies/metadata.csv remains the combined anomaly metadata,
    without destroying S1.
    """

    metadata_path = out_dir / "metadata.csv"

    if metadata_path.exists():
        old = pd.read_csv(metadata_path)

        if "scenario" in old.columns:
            old = old[~old["scenario"].astype(str).isin(S2_SUBTYPES)].copy()
        else:
            # If the old metadata has no scenario column, keep it untouched.
            old = old.copy()

        new = pd.DataFrame(s2_rows)

        # Union of columns, preserving old columns first.
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
    print("[Q3 S2 DIAGNOSTICS]")
    print("=" * 80)

    print(f"\n[TOTAL S2] n={len(df)}")
    print(f"  scenarios={df['scenario'].value_counts().to_dict()}")

    print("\n[PARAMS FROM NORMAL DISTRIBUTION]")
    print(
        f"  manufact unique={df['manufact'].nunique()} "
        f"min={int(df['manufact'].min())} "
        f"max={int(df['manufact'].max())} "
        f"mean={df['manufact'].mean():.2f} "
        f"std={df['manufact'].std():.2f}"
    )
    print(f"  month unique={sorted(df['month'].unique().astype(int).tolist())}")

    print("\n[ROWS]")
    summary = (
        df.groupby("scenario")
        .agg(
            n=("file", "count"),
            normal_min=("normal_n_rows", "min"),
            normal_median=("normal_n_rows", "median"),
            normal_max=("normal_n_rows", "max"),
            s2_min=("s2_n_rows", "min"),
            s2_median=("s2_n_rows", "median"),
            s2_max=("s2_n_rows", "max"),
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
            "Generate Q3 S2 anomalies only. "
            "Read normal metadata and write SQL files and S2 metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--db", "--db-path", dest="db_path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--schema", default=DB_SCHEMA, help="Database search path.")
    p.add_argument("--normal-metadata", type=Path, required=True, help="Metadata CSV of the normal Q3 workload.")

    output = p.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output-dir", type=Path,
        help="Parent directory; output is written to its q3_s2_anomalies subfolder.",
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

    args = p.parse_args(argv)
    args.db_path = args.db_path.expanduser().resolve()
    args.normal_metadata = args.normal_metadata.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
        if args.output_dir.exists() and not args.output_dir.is_dir():
            p.error(f"Output path is not a directory: {args.output_dir}")
        args.out_anom = args.output_dir / "q3_s2_anomalies"
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=" * 80)
    print("[Q3 S2 QUERY GENERATOR]")
    print("=" * 80)
    print(f"DB_PATH={args.db_path}")
    print(f"DB_SCHEMA={args.schema}")
    print(f"NORMAL_METADATA={args.normal_metadata}")
    print(f"OUT_ANOM={args.out_anom}")
    print(f"SEED={args.seed}")
    print(f"N_ANOMALY_PER_TYPE={args.n_per_type}")
    print(f"S2_SUBTYPES={S2_SUBTYPES}")
    print("=" * 80)

    normal_df = load_normal_metadata(args.normal_metadata)

    print(f"\n[NORMAL METADATA] rows={len(normal_df)}")
    print(
        f"  manufact min={int(normal_df['manufact'].min())} "
        f"max={int(normal_df['manufact'].max())} "
        f"unique={normal_df['manufact'].nunique()}"
    )
    print(f"  month unique={sorted(normal_df['month'].unique().astype(int).tolist())}")

    all_rows: list[dict] = []
    con = duckdb.connect(str(args.db_path), read_only=True)
    try:
        con.execute("SET search_path = ?", [args.schema])
        if args.clean_s2:
            print("\n[CLEAN] removing old S2 files only")
            remove_old_s2_files(args.out_anom)

        rng = np.random.default_rng(args.seed)
        for subtype in S2_SUBTYPES:
            print(f"\n[GEN] {subtype}")
            rows = write_s2_queries(
                out_dir=args.out_anom,
                subtype=subtype,
                n=args.n_per_type,
                normal_df=normal_df,
                con=con,
                rng=rng,
            )
            all_rows.extend(rows)
    finally:
        con.close()

    metadata_s2 = args.out_anom / "metadata_s2.csv"
    write_metadata(metadata_s2, all_rows)
    print(f"\n[WRITE] {metadata_s2} rows={len(all_rows)}")

    q3_s2_metadata = args.out_anom / "q3_s2_metadata.csv"
    write_metadata(q3_s2_metadata, all_rows)
    print(f"[WRITE] {q3_s2_metadata} rows={len(all_rows)}")

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

    summary_path = args.out_anom / "q3_s2_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[WRITE] {summary_path} rows={len(summary)}")

    if not args.no_merge_metadata:
        merge_metadata(args.out_anom, all_rows)

    print_diagnostics(all_rows)

    print("\n[DONE]")


if __name__ == "__main__":
    main()

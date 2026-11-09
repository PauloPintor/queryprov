#!/usr/bin/env python3
from __future__ import annotations

import argparse

import csv
from pathlib import Path
from typing import Dict, Optional

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DB_PATH = None  # Set by command-line arguments.
DB_SCHEMA = "tpcds"

SEED = 42

# Gerar mais do que o necessário para permitir sampling posterior.
# 500 por subtipo -> 3000 S2 estruturais para Q60.
N_S2_PER_TYPE = 500

Q60_NORMAL_DIR = None  # Set by command-line arguments.

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.
OUT_S2 = None  # Set by command-line arguments.

# Fallback, caso não exista metadata.csv dos normais.
YEAR = 2002
MONTHS = [8, 9, 10]

FALLBACK_CATEGORIES = [
    "Children",
    "Men",
    "Music",
    "Jewelry",
    "Shoes",
]

FALLBACK_GMTS = [
    -8.0,
    -7.0,
    -6.0,
    -5.0,
]

S2_SUBTYPES = [
    "q60_s2_join_add_customer",
    "q60_s2_where_remove_month",
    "q60_s2_agg_no_group_by",
    "q60_s2_agg_group_by_channel",
    "q60_s2_limit_1",
    "q60_s2_limit_none",
]

S2_INFO: Dict[str, dict] = {
    "q60_s2_join_add_customer": {
        "label": 2,
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "intent_preservation": "high",
        "description": (
            "Adds the customer table to each sales-channel branch through a valid "
            "customer-key join. The original category, GMT, year and month "
            "parameters are preserved, but each witness gains customer provenance."
        ),
    },
    "q60_s2_where_remove_month": {
        "label": 2,
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "intent_preservation": "medium",
        "description": (
            "Removes the contextual month predicate from each sales-channel branch. "
            "The S1 dimensions category and GMT are preserved, so the mutation "
            "does not collapse the parametric profile of Q60."
        ),
    },
    "q60_s2_agg_no_group_by": {
        "label": 2,
        "category": "s2_agg",
        "logical_unit": "aggregation granularity",
        "intent_preservation": "medium",
        "description": (
            "Collapses the final item-level aggregation to a single global "
            "aggregate. Parameters and branch predicates remain normal."
        ),
    },
    "q60_s2_agg_group_by_channel": {
        "label": 2,
        "category": "s2_agg",
        "logical_unit": "aggregation granularity",
        "intent_preservation": "medium",
        "description": (
            "Changes the final aggregation granularity from item-level to "
            "sales-channel-level, preserving the three-channel structure."
        ),
    },
    "q60_s2_limit_1": {
        "label": 2,
        "category": "s2_limit",
        "logical_unit": "result cardinality limit",
        "intent_preservation": "medium",
        "description": (
            "Changes the final LIMIT 100 to LIMIT 1, restricting the materialized "
            "result while preserving the query structure and normal parameters."
        ),
    },
    "q60_s2_limit_none": {
        "label": 2,
        "category": "s2_limit",
        "logical_unit": "result cardinality limit",
        "intent_preservation": "medium",
        "description": (
            "Removes the final LIMIT 100. Diagnostics showed that Q60 is normally "
            "truncated by the limit, so this produces an observable expansion "
            "without changing joins, filters or sampled parameters."
        ),
    },
}


# ============================================================
# PARAMETER SAMPLING
# ============================================================

def load_normal_metadata(path: Path) -> Optional[pd.DataFrame]:
    meta_path = path / "metadata.csv"

    if not meta_path.exists():
        return None

    df = pd.read_csv(meta_path)

    required = [
        "category",
        "gmt",
        "year",
        "month",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"[WARN] metadata.csv missing columns: {missing}")
        return None

    df = df.dropna(subset=required).copy()

    if df.empty:
        return None

    df["category"] = df["category"].astype(str)
    df["gmt"] = df["gmt"].astype(float)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    return df


class Q60S2Sampler:
    def __init__(
        self,
        normal_metadata_dir: Path,
        seed: int,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.normal_meta = load_normal_metadata(normal_metadata_dir)

        if self.normal_meta is not None:
            print(f"[PARAMS] using normal metadata: {normal_metadata_dir / 'metadata.csv'}")
            print(f"[PARAMS] normal metadata rows={len(self.normal_meta):,}")
        else:
            print("[PARAMS] normal metadata unavailable; using fallback sampler")

    def sample_params(self) -> dict:
        if self.normal_meta is None:
            return {
                "param_source": "fallback_sampler",
                "category": str(self.rng.choice(FALLBACK_CATEGORIES)),
                "gmt": float(self.rng.choice(FALLBACK_GMTS)),
                "year": YEAR,
                "month": int(self.rng.choice(MONTHS)),
            }

        idx = int(
            self.rng.choice(
                self.normal_meta.index.to_numpy(),
                size=1,
                replace=True,
            )[0]
        )

        row = self.normal_meta.loc[idx]

        return {
            "param_source": "normal_metadata",
            "category": str(row["category"]),
            "gmt": float(row["gmt"]),
            "year": int(row["year"]),
            "month": int(row["month"]),
        }

    def sample(self, subtype: str) -> dict:
        info = S2_INFO[subtype]
        params = self.sample_params()

        params.update(
            {
                "label": int(info["label"]),
                "scenario": subtype,
                "category_s2": info["category"],
                "logical_unit": info["logical_unit"],
                "intent_preservation": info["intent_preservation"],
                "description": info["description"],
            }
        )

        return params


# ============================================================
# SQL HELPERS
# ============================================================

def sql_quote(s: str) -> str:
    return str(s).replace("'", "''")


def format_gmt(gmt: float) -> str:
    if float(gmt).is_integer():
        return str(int(gmt))

    return str(float(gmt))


def prepare_params(params: dict) -> dict:
    p = dict(params)

    p["category"] = sql_quote(str(p["category"]))
    p["gmt"] = format_gmt(float(p["gmt"]))
    p["year"] = int(p["year"])
    p["month"] = int(p["month"])

    return p


def render_sql(
    template: str,
    params: dict,
) -> str:
    return template.format(**prepare_params(params))


# ============================================================
# SQL BUILDER
# ============================================================

def month_predicate(subtype: str) -> str:
    if subtype == "q60_s2_where_remove_month":
        return ""

    return "      AND d.d_moy = {month}\n"


def use_customer_join(subtype: str) -> bool:
    return subtype == "q60_s2_join_add_customer"


def included_channels(subtype: str) -> list[str]:
    # No final S2 generator we preserve the three-channel identity of Q60.
    return ["ss", "cs", "ws"]


def branch_config(channel: str) -> dict:
    if channel == "ss":
        return {
            "cte": "ss",
            "table": "store_sales",
            "alias": "s",
            "sales_col": "ss_ext_sales_price",
            "item_key": "ss_item_sk",
            "date_key": "ss_sold_date_sk",
            "addr_key": "ss_addr_sk",
            "customer_key": "ss_customer_sk",
            "channel_name": "store",
        }

    if channel == "cs":
        return {
            "cte": "cs",
            "table": "catalog_sales",
            "alias": "s",
            "sales_col": "cs_ext_sales_price",
            "item_key": "cs_item_sk",
            "date_key": "cs_sold_date_sk",
            "addr_key": "cs_bill_addr_sk",
            "customer_key": "cs_bill_customer_sk",
            "channel_name": "catalog",
        }

    if channel == "ws":
        return {
            "cte": "ws",
            "table": "web_sales",
            "alias": "s",
            "sales_col": "ws_ext_sales_price",
            "item_key": "ws_item_sk",
            "date_key": "ws_sold_date_sk",
            "addr_key": "ws_bill_addr_sk",
            "customer_key": "ws_bill_customer_sk",
            "channel_name": "web",
        }

    raise ValueError(f"Unknown channel: {channel}")


def build_branch(
    channel: str,
    subtype: str,
    with_prov: bool,
) -> str:
    cfg = branch_config(channel)

    table = cfg["table"]
    alias = cfg["alias"]
    sales_col = cfg["sales_col"]
    item_key = cfg["item_key"]
    date_key = cfg["date_key"]
    addr_key = cfg["addr_key"]
    customer_key = cfg["customer_key"]
    channel_name = cfg["channel_name"]

    extra_from = ""
    customer_join = ""
    prov_terms = f"{alias}.prov, d.prov, ca.prov, i.prov"

    if use_customer_join(subtype):
        extra_from = ",\n         customer c"
        customer_join = f"      AND {alias}.{customer_key} = c.c_customer_sk\n"
        prov_terms = f"{alias}.prov, d.prov, ca.prov, i.prov, c.prov"

    if with_prov:
        prov_select = f"""
        array_agg(
            list_value(
                {prov_terms}
            )
        ) AS prov
"""
    else:
        prov_select = ""

    channel_select = ""

    if subtype == "q60_s2_agg_group_by_channel":
        channel_select = f"        '{channel_name}' AS channel,\n"

    select_parts = f"""
    SELECT
{channel_select}        i.i_item_id,
        sum({alias}.{sales_col}) AS total_sales"""

    if with_prov:
        select_parts += "," + prov_select

    sql = f"""
{cfg["cte"]} AS (
{select_parts}
    FROM {table} {alias},
         date_dim d,
         customer_address ca,
         item i
{extra_from}
    WHERE i.i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('{{category}}')
          )
      AND {alias}.{item_key} = i.i_item_sk
      AND {alias}.{date_key} = d.d_date_sk
      AND d.d_year = {{year}}
{month_predicate(subtype)}      AND {alias}.{addr_key} = ca.ca_address_sk
      AND ca.ca_gmt_offset = {{gmt}}
{customer_join}    GROUP BY i.i_item_id
)
"""

    return sql


def build_ctes(
    subtype: str,
    with_prov: bool,
) -> str:
    branches = [
        build_branch(
            channel=ch,
            subtype=subtype,
            with_prov=with_prov,
        )
        for ch in included_channels(subtype)
    ]

    return "WITH " + ",\n".join(branches)


def build_union(subtype: str) -> str:
    parts = []

    for ch in included_channels(subtype):
        cfg = branch_config(ch)
        parts.append(f"    SELECT * FROM {cfg['cte']}")

    return "\n\n    UNION ALL\n\n".join(parts)


def final_limit_clause(subtype: str) -> str:
    if subtype == "q60_s2_limit_1":
        return "LIMIT 1"

    if subtype == "q60_s2_limit_none":
        return ""

    return "LIMIT 100"


def build_full_sql(subtype: str) -> str:
    ctes = build_ctes(subtype, with_prov=True)
    union = build_union(subtype)

    if subtype == "q60_s2_agg_no_group_by":
        return f"""
{ctes}

SELECT
    sum(total_sales) AS total_sales,
    list_reduce(
        array_agg(prov),
        (x, y) -> list_concat(x, y)
    ) AS prov
FROM (
{union}
) tmp1
"""

    if subtype == "q60_s2_agg_group_by_channel":
        return f"""
{ctes}

SELECT
    channel,
    sum(total_sales) AS total_sales,
    list_reduce(
        array_agg(prov),
        (x, y) -> list_concat(x, y)
    ) AS prov
FROM (
{union}
) tmp1
GROUP BY channel
ORDER BY channel
LIMIT 100
"""

    return f"""
{ctes}

SELECT
    i_item_id,
    sum(total_sales) AS total_sales,
    list_reduce(
        array_agg(prov),
        (x, y) -> list_concat(x, y)
    ) AS prov
FROM (
{union}
) tmp1
GROUP BY i_item_id
ORDER BY
    i_item_id,
    total_sales
{final_limit_clause(subtype)}
"""


def build_count_sql(
    subtype: str,
    materialized_limit: bool,
) -> str:
    ctes = build_ctes(subtype, with_prov=False)
    union = build_union(subtype)

    if subtype == "q60_s2_agg_no_group_by":
        return f"""
{ctes}

SELECT count(*) AS n_rows
FROM (
    SELECT
        sum(total_sales) AS total_sales
    FROM (
{union}
    ) tmp1
) g
WHERE total_sales IS NOT NULL
"""

    if subtype == "q60_s2_agg_group_by_channel":
        limit = "LIMIT 100" if materialized_limit else ""

        return f"""
{ctes}

SELECT count(*) AS n_rows
FROM (
    SELECT
        channel
    FROM (
{union}
    ) tmp1
    GROUP BY channel
    ORDER BY channel
    {limit}
) g
"""

    limit = ""

    if materialized_limit:
        limit = final_limit_clause(subtype)

    return f"""
{ctes}

SELECT count(*) AS n_rows
FROM (
    SELECT
        i_item_id
    FROM (
{union}
    ) tmp1
    GROUP BY i_item_id
    ORDER BY
        i_item_id
    {limit}
) g
"""


# ============================================================
# VALIDATION
# ============================================================

def validate_params(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    params: dict,
) -> tuple[bool, dict]:
    """
    Keep only executable, non-empty S2 instances.

    We also record limited and unbounded cardinalities. For limit_none,
    the observed query is intentionally unbounded; for normal-like variants,
    limited_nrows corresponds to the materialized result.
    """

    limited_sql = build_count_sql(
        subtype,
        materialized_limit=True,
    )

    unbounded_sql = build_count_sql(
        subtype,
        materialized_limit=False,
    )

    limited_nrows = con.execute(
        render_sql(limited_sql, params)
    ).fetchone()[0]

    unbounded_nrows = con.execute(
        render_sql(unbounded_sql, params)
    ).fetchone()[0]

    limited_nrows = int(limited_nrows)
    unbounded_nrows = int(unbounded_nrows)

    # For q60_s2_limit_none, the materialized result is unbounded.
    if subtype == "q60_s2_limit_none":
        effective_nrows = unbounded_nrows
    else:
        effective_nrows = limited_nrows

    valid = effective_nrows > 0

    meta = {
        "n_rows": int(effective_nrows),
        "limited_nrows": int(limited_nrows),
        "unbounded_nrows": int(unbounded_nrows),
    }

    return valid, meta


def is_duplicate_sql(
    sql: str,
    seen: set[str],
) -> bool:
    key = " ".join(sql.split())

    if key in seen:
        return True

    seen.add(key)
    return False


# ============================================================
# WRITER
# ============================================================

def write_metadata(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    fields = sorted(
        {
            k
            for row in rows
            for k in row.keys()
        }
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_queries_for_subtype(
    subtype: str,
    n: int,
    sampler: Q60S2Sampler,
    con: duckdb.DuckDBPyConnection,
    global_rows: list[dict],
) -> None:
    OUT_S2.mkdir(parents=True, exist_ok=True)

    sql_template = build_full_sql(subtype)

    short = subtype.replace("q60_s2_", "")
    short = short.replace("_", "")

    seen_sql: set[str] = set()

    i = 0
    attempts = 0

    max_attempts = max(100_000, n * 500)

    print(f"\n[GEN] {subtype}")
    print(f"      target={n}")

    while i < n:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Too many attempts for {subtype}: "
                f"generated {i}/{n} after {attempts} attempts."
            )

        params = sampler.sample(subtype)

        try:
            valid, count_meta = validate_params(
                con=con,
                subtype=subtype,
                params=params,
            )
        except Exception as e:
            print(f"[WARN] validation failed for {subtype}: {e}")
            continue

        if not valid:
            continue

        sql = render_sql(sql_template, params).strip() + "\n"

        if is_duplicate_sql(sql, seen_sql):
            continue

        filename = f"{short}_{i:05d}.sql"

        (OUT_S2 / filename).write_text(
            sql,
            encoding="utf-8",
        )

        row = {
            "file": filename,
            **params,
            **count_meta,
        }

        global_rows.append(row)

        i += 1

        if i % 100 == 0:
            print(f"  [GEN] {subtype}: {i}/{n}")

    print(
        f"  [OK] {subtype}: {i} queries, attempts={attempts}"
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_param_summary(
    df: pd.DataFrame,
    label: str,
) -> None:
    print(f"\n[{label} PARAMS] n={len(df):,}")

    if df.empty:
        return

    print(
        f"  category unique={df['category'].nunique()} "
        f"values={sorted(df['category'].dropna().unique().tolist())[:10]}"
    )

    print(
        f"  gmt      unique={df['gmt'].nunique()} "
        f"min={float(df['gmt'].min()):.2f} "
        f"max={float(df['gmt'].max()):.2f}"
    )

    print(
        f"  year     unique={df['year'].nunique()} "
        f"values={sorted(df['year'].dropna().astype(int).unique().tolist())}"
    )

    print(
        f"  month    unique={df['month'].nunique()} "
        f"values={sorted(df['month'].dropna().astype(int).unique().tolist())}"
    )


def print_count_summary(
    df: pd.DataFrame,
    label: str,
) -> None:
    print(f"\n[{label} COUNTS]")

    for c in [
        "n_rows",
        "limited_nrows",
        "unbounded_nrows",
    ]:
        if c not in df.columns:
            continue

        s = pd.to_numeric(
            df[c],
            errors="coerce",
        ).dropna()

        if s.empty:
            continue

        print(
            f"  {c:<16} "
            f"min={s.min():.0f} "
            f"median={s.median():.0f} "
            f"max={s.max():.0f}"
        )


def print_diagnostics(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("[DIAGNOSTICS]")
    print("=" * 80)

    print(f"\n[TOTAL] {len(df):,} S2 queries")
    print(f"[SCENARIOS] {df['scenario'].value_counts().to_dict()}")

    print_param_summary(df, "ALL S2")

    for subtype in S2_SUBTYPES:
        g = df[df["scenario"] == subtype]

        if g.empty:
            continue

        print("\n" + "-" * 80)
        print(f"[{subtype}] n={len(g):,}")
        print(f"  category={S2_INFO[subtype]['category']}")
        print(f"  logical_unit={S2_INFO[subtype]['logical_unit']}")
        print(f"  intent_preservation={S2_INFO[subtype]['intent_preservation']}")

        print_param_summary(g, subtype)
        print_count_summary(g, subtype)

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q60 S2 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q60.")
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
    for name in ['q60_s2_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, EXTERNAL_BASE_DIR, N_S2_PER_TYPE, OUT_S2, Q60_NORMAL_DIR, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_S2_PER_TYPE = args.n_per_type
    OUT_S2 = args.output_dir / "q60_s2_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir
    Q60_NORMAL_DIR = args.normal_dir


def main() -> None:
    configure_cli()
    print("=" * 90)
    print("[Q60 S2 GENERATOR]")
    print("=" * 90)
    print(f"DB_PATH={DB_PATH}")
    print(f"DB_SCHEMA={DB_SCHEMA}")
    print(f"Q60_NORMAL_DIR={Q60_NORMAL_DIR}")
    print(f"OUT_S2={OUT_S2}")
    print(f"N_S2_PER_TYPE={N_S2_PER_TYPE}")
    print(f"S2_SUBTYPES={S2_SUBTYPES}")
    print("=" * 90)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET search_path = ?", [DB_SCHEMA])

    sampler = Q60S2Sampler(
        normal_metadata_dir=Q60_NORMAL_DIR,
        seed=SEED,
    )

    OUT_S2.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for subtype in S2_SUBTYPES:
        write_queries_for_subtype(
            subtype=subtype,
            n=N_S2_PER_TYPE,
            sampler=sampler,
            con=con,
            global_rows=all_rows,
        )

    write_metadata(
        OUT_S2 / "metadata.csv",
        all_rows,
    )

    con.close()

    print_diagnostics(all_rows)

    print("\n[DONE]")
    print(f"  queries   : {len(all_rows):,}")
    print(f"  directory : {OUT_S2}")
    print(f"  metadata  : {OUT_S2 / 'metadata.csv'}")


if __name__ == "__main__":
    main()
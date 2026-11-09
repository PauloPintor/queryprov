#!/usr/bin/env python3
from __future__ import annotations

import argparse

import csv
from pathlib import Path
from typing import Callable

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DB_PATH = None  # Set by command-line arguments.
DB_SCHEMA = "tpcds"

SEED = 42

N_NORMAL = 65_000
N_ANOMALY_PER_TYPE = 350

GENERATE_MIX = True

# K controla a concentração dos pesos dentro dos pools.
# K=0.5 -> mais concentrado nos valores frequentes
# K=1.0 -> pesos proporcionais à frequência
# K>1   -> mais uniforme
K = 1.0

# Este é o substituto prático do Q75.
# Se quiseres manter outro identificador, muda só QUERY e EXTERNAL_BASE_DIR.
QUERY = "q60"

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.


# ============================================================
# FIXED / NORMAL CONTEXT
# ============================================================

YEAR = 2002

# O template original usa 8..10.
# Aqui os meses são sempre normais/contextuais, não são dimensão anómala.
MONTHS = [8, 9, 10]

# Categorias do template original.
CATEGORIES = [
    "Children",
    "Men",
    "Music",
    "Jewelry",
    "Shoes",
]


# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
WITH ss AS (
    SELECT
        i_item_id,
        sum(ss_ext_sales_price) AS total_sales,
        array_agg(
            list_value(
                store_sales.prov,
                date_dim.prov,
                customer_address.prov,
                item.prov
            )
        ) AS prov
    FROM store_sales,
         date_dim,
         customer_address,
         item
    WHERE i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('{category}')
          )
      AND ss_item_sk = i_item_sk
      AND ss_sold_date_sk = d_date_sk
      AND d_year = {year}
      AND d_moy = {month}
      AND ss_addr_sk = ca_address_sk
      AND ca_gmt_offset = {gmt}
    GROUP BY i_item_id
),

cs AS (
    SELECT
        i_item_id,
        sum(cs_ext_sales_price) AS total_sales,
        array_agg(
            list_value(
                catalog_sales.prov,
                date_dim.prov,
                customer_address.prov,
                item.prov
            )
        ) AS prov
    FROM catalog_sales,
         date_dim,
         customer_address,
         item
    WHERE i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('{category}')
          )
      AND cs_item_sk = i_item_sk
      AND cs_sold_date_sk = d_date_sk
      AND d_year = {year}
      AND d_moy = {month}
      AND cs_bill_addr_sk = ca_address_sk
      AND ca_gmt_offset = {gmt}
    GROUP BY i_item_id
),

ws AS (
    SELECT
        i_item_id,
        sum(ws_ext_sales_price) AS total_sales,
        array_agg(
            list_value(
                web_sales.prov,
                date_dim.prov,
                customer_address.prov,
                item.prov
            )
        ) AS prov
    FROM web_sales,
         date_dim,
         customer_address,
         item
    WHERE i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('{category}')
          )
      AND ws_item_sk = i_item_sk
      AND ws_sold_date_sk = d_date_sk
      AND d_year = {year}
      AND d_moy = {month}
      AND ws_bill_addr_sk = ca_address_sk
      AND ca_gmt_offset = {gmt}
    GROUP BY i_item_id
)

SELECT
    i_item_id,
    sum(total_sales) AS total_sales,
    list_reduce(
        array_agg(prov),
        (x, y) -> list_concat(x, y)
    ) AS prov
FROM (
    SELECT *
    FROM ss

    UNION ALL

    SELECT *
    FROM cs

    UNION ALL

    SELECT *
    FROM ws
) tmp1
GROUP BY i_item_id
ORDER BY
    i_item_id,
    total_sales
LIMIT 100
"""


COUNT_TEMPLATE = """
WITH ss AS (
    SELECT
        i_item_id,
        sum(ss_ext_sales_price) AS total_sales
    FROM store_sales,
         date_dim,
         customer_address,
         item
    WHERE i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('{category}')
          )
      AND ss_item_sk = i_item_sk
      AND ss_sold_date_sk = d_date_sk
      AND d_year = {year}
      AND d_moy = {month}
      AND ss_addr_sk = ca_address_sk
      AND ca_gmt_offset = {gmt}
    GROUP BY i_item_id
),

cs AS (
    SELECT
        i_item_id,
        sum(cs_ext_sales_price) AS total_sales
    FROM catalog_sales,
         date_dim,
         customer_address,
         item
    WHERE i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('{category}')
          )
      AND cs_item_sk = i_item_sk
      AND cs_sold_date_sk = d_date_sk
      AND d_year = {year}
      AND d_moy = {month}
      AND cs_bill_addr_sk = ca_address_sk
      AND ca_gmt_offset = {gmt}
    GROUP BY i_item_id
),

ws AS (
    SELECT
        i_item_id,
        sum(ws_ext_sales_price) AS total_sales
    FROM web_sales,
         date_dim,
         customer_address,
         item
    WHERE i_item_id IN (
            SELECT i_item_id
            FROM item
            WHERE i_category IN ('{category}')
          )
      AND ws_item_sk = i_item_sk
      AND ws_sold_date_sk = d_date_sk
      AND d_year = {year}
      AND d_moy = {month}
      AND ws_bill_addr_sk = ca_address_sk
      AND ca_gmt_offset = {gmt}
    GROUP BY i_item_id
)

SELECT count(*) AS n_rows
FROM (
    SELECT
        i_item_id
    FROM (
        SELECT *
        FROM ss

        UNION ALL

        SELECT *
        FROM cs

        UNION ALL

        SELECT *
        FROM ws
    ) tmp1
    GROUP BY i_item_id
) g
"""


# ============================================================
# FITTING CATEGORICAL POOLS
# ============================================================

def _normalise_weights(counts: np.ndarray, k: float) -> np.ndarray:
    alpha = 1.0 / max(k, 0.05)

    weights = counts.astype(float) ** alpha
    weights = weights / weights.sum()

    return weights


def fit_category_tiers(
    con: duckdb.DuckDBPyConnection,
    k: float,
    rng: np.random.Generator,
) -> dict:
    """
    Frequências de CATEGORY no espaço executável do template.

    YEAR fixo em 2002.
    MONTHS = 8, 9, 10.
    GMT livre, porque GMT é outra dimensão.
    """

    categories_sql = ", ".join(f"'{c.replace(chr(39), chr(39) + chr(39))}'" for c in CATEGORIES)
    months_sql = ", ".join(str(m) for m in MONTHS)

    df = con.execute(f"""
        WITH all_sales AS (
            SELECT
                i.i_category AS category,
                COUNT(*) AS n_rows
            FROM store_sales ss,
                 date_dim d,
                 customer_address ca,
                 item i
            WHERE ss.ss_item_sk = i.i_item_sk
              AND ss.ss_sold_date_sk = d.d_date_sk
              AND ss.ss_addr_sk = ca.ca_address_sk
              AND d.d_year = {YEAR}
              AND d.d_moy IN ({months_sql})
              AND i.i_category IN ({categories_sql})
              AND ca.ca_gmt_offset IS NOT NULL
            GROUP BY i.i_category

            UNION ALL

            SELECT
                i.i_category AS category,
                COUNT(*) AS n_rows
            FROM catalog_sales cs,
                 date_dim d,
                 customer_address ca,
                 item i
            WHERE cs.cs_item_sk = i.i_item_sk
              AND cs.cs_sold_date_sk = d.d_date_sk
              AND cs.cs_bill_addr_sk = ca.ca_address_sk
              AND d.d_year = {YEAR}
              AND d.d_moy IN ({months_sql})
              AND i.i_category IN ({categories_sql})
              AND ca.ca_gmt_offset IS NOT NULL
            GROUP BY i.i_category

            UNION ALL

            SELECT
                i.i_category AS category,
                COUNT(*) AS n_rows
            FROM web_sales ws,
                 date_dim d,
                 customer_address ca,
                 item i
            WHERE ws.ws_item_sk = i.i_item_sk
              AND ws.ws_sold_date_sk = d.d_date_sk
              AND ws.ws_bill_addr_sk = ca.ca_address_sk
              AND d.d_year = {YEAR}
              AND d.d_moy IN ({months_sql})
              AND i.i_category IN ({categories_sql})
              AND ca.ca_gmt_offset IS NOT NULL
            GROUP BY i.i_category
        )
        SELECT
            category,
            SUM(n_rows) AS n_rows
        FROM all_sales
        GROUP BY category
        ORDER BY n_rows DESC, category
    """).df()

    if len(df) < 3:
        raise ValueError(f"Poucas categorias executáveis: {len(df)}")

    values = df["category"].astype(str).tolist()
    counts = df["n_rows"].astype(float).to_numpy()

    n = len(values)

    # Com 5 categorias: normal=3, anomaly=2.
    # Generalização defensiva para outros casos.
    normal_size = int(np.ceil(n / 2.0))
    normal_size = max(1, min(normal_size, n - 1))

    base_probs = counts / counts.sum()

    normal_idx = rng.choice(
        np.arange(n),
        size=normal_size,
        replace=False,
        p=base_probs,
    )

    normal_idx = sorted(int(i) for i in normal_idx)
    normal_set = set(normal_idx)

    anomaly_idx = [
        i for i in range(n)
        if i not in normal_set
    ]

    normal_values = [values[i] for i in normal_idx]
    anomaly_values = [values[i] for i in anomaly_idx]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    normal_w = _normalise_weights(normal_counts, k)
    anomaly_w = _normalise_weights(anomaly_counts, k)

    print("\n[CATEGORY]")
    print(f"  n_categories={n}  K={k}")
    print(f"  normal n={len(normal_values)}  anomaly n={len(anomaly_values)}")
    print(f"  normal={normal_values}")
    print(f"  anomaly={anomaly_values}")
    print("\n[CATEGORY FREQUENCIES]")
    print(df.to_string(index=False))

    return {
        "normal": normal_values,
        "normal_w": normal_w,
        "anomaly": anomaly_values,
        "anomaly_w": anomaly_w,
        "df": df,
    }


def fit_gmt_tiers(
    con: duckdb.DuckDBPyConnection,
    k: float,
    rng: np.random.Generator,
) -> dict:
    """
    Frequências de GMT no espaço executável do template.

    YEAR fixo em 2002.
    MONTHS = 8, 9, 10.
    CATEGORY livre entre as categorias do template, porque CATEGORY é outra dimensão.
    """

    categories_sql = ", ".join(f"'{c.replace(chr(39), chr(39) + chr(39))}'" for c in CATEGORIES)
    months_sql = ", ".join(str(m) for m in MONTHS)

    df = con.execute(f"""
        WITH all_sales AS (
            SELECT
                ca.ca_gmt_offset AS gmt,
                COUNT(*) AS n_rows
            FROM store_sales ss,
                 date_dim d,
                 customer_address ca,
                 item i
            WHERE ss.ss_item_sk = i.i_item_sk
              AND ss.ss_sold_date_sk = d.d_date_sk
              AND ss.ss_addr_sk = ca.ca_address_sk
              AND d.d_year = {YEAR}
              AND d.d_moy IN ({months_sql})
              AND i.i_category IN ({categories_sql})
              AND ca.ca_gmt_offset IS NOT NULL
            GROUP BY ca.ca_gmt_offset

            UNION ALL

            SELECT
                ca.ca_gmt_offset AS gmt,
                COUNT(*) AS n_rows
            FROM catalog_sales cs,
                 date_dim d,
                 customer_address ca,
                 item i
            WHERE cs.cs_item_sk = i.i_item_sk
              AND cs.cs_sold_date_sk = d.d_date_sk
              AND cs.cs_bill_addr_sk = ca.ca_address_sk
              AND d.d_year = {YEAR}
              AND d.d_moy IN ({months_sql})
              AND i.i_category IN ({categories_sql})
              AND ca.ca_gmt_offset IS NOT NULL
            GROUP BY ca.ca_gmt_offset

            UNION ALL

            SELECT
                ca.ca_gmt_offset AS gmt,
                COUNT(*) AS n_rows
            FROM web_sales ws,
                 date_dim d,
                 customer_address ca,
                 item i
            WHERE ws.ws_item_sk = i.i_item_sk
              AND ws.ws_sold_date_sk = d.d_date_sk
              AND ws.ws_bill_addr_sk = ca.ca_address_sk
              AND d.d_year = {YEAR}
              AND d.d_moy IN ({months_sql})
              AND i.i_category IN ({categories_sql})
              AND ca.ca_gmt_offset IS NOT NULL
            GROUP BY ca.ca_gmt_offset
        )
        SELECT
            gmt,
            SUM(n_rows) AS n_rows
        FROM all_sales
        GROUP BY gmt
        ORDER BY n_rows DESC, gmt
    """).df()

    if len(df) < 4:
        raise ValueError(f"Poucos GMT offsets executáveis: {len(df)}")

    values = df["gmt"].astype(float).tolist()
    counts = df["n_rows"].astype(float).to_numpy()

    n = len(values)

    normal_size = n // 2
    normal_size = max(1, min(normal_size, n - 1))

    base_probs = counts / counts.sum()

    normal_idx = rng.choice(
        np.arange(n),
        size=normal_size,
        replace=False,
        p=base_probs,
    )

    normal_idx = sorted(int(i) for i in normal_idx)
    normal_set = set(normal_idx)

    anomaly_idx = [
        i for i in range(n)
        if i not in normal_set
    ]

    normal_values = [values[i] for i in normal_idx]
    anomaly_values = [values[i] for i in anomaly_idx]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    normal_w = _normalise_weights(normal_counts, k)
    anomaly_w = _normalise_weights(anomaly_counts, k)

    print("\n[GMT]")
    print(f"  n_gmt={n}  K={k}")
    print(f"  normal n={len(normal_values)}  anomaly n={len(anomaly_values)}")
    print(f"  normal={normal_values}")
    print(f"  anomaly={anomaly_values}")
    print("\n[GMT FREQUENCIES]")
    print(df.to_string(index=False))

    return {
        "normal": normal_values,
        "normal_w": normal_w,
        "anomaly": anomaly_values,
        "anomaly_w": anomaly_w,
        "df": df,
    }


def sample_weighted(
    pool: list,
    weights: np.ndarray,
    rng: np.random.Generator,
):
    if not pool:
        raise ValueError("Pool vazio.")

    idx = int(
        rng.choice(
            len(pool),
            size=1,
            replace=True,
            p=weights,
        )[0]
    )

    return pool[idx]


def sample_month(rng: np.random.Generator) -> int:
    return int(
        rng.choice(
            MONTHS,
            size=1,
            replace=True,
        )[0]
    )


# ============================================================
# SAMPLER
# ============================================================

class QuerySampler:
    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        rng: np.random.Generator,
    ):
        self.rng = rng

        self.category = fit_category_tiers(
            con,
            K,
            rng,
        )

        self.gmt = fit_gmt_tiers(
            con,
            K,
            rng,
        )

        print("\n[CONTEXT]")
        print(f"  YEAR={YEAR}")
        print(f"  MONTHS={MONTHS}")
        print("  year/month are contextual normal parameters, not anomaly dimensions")

    def normal_params(self) -> dict:
        category = sample_weighted(
            self.category["normal"],
            self.category["normal_w"],
            self.rng,
        )

        gmt = sample_weighted(
            self.gmt["normal"],
            self.gmt["normal_w"],
            self.rng,
        )

        month = sample_month(self.rng)

        return self._params(
            scenario="normal",
            label=0,
            category=category,
            gmt=gmt,
            month=month,
        )

    def s1_category_params(self) -> dict:
        category = sample_weighted(
            self.category["anomaly"],
            self.category["anomaly_w"],
            self.rng,
        )

        gmt = sample_weighted(
            self.gmt["normal"],
            self.gmt["normal_w"],
            self.rng,
        )

        month = sample_month(self.rng)

        return self._params(
            scenario="s1_category",
            label=1,
            category=category,
            gmt=gmt,
            month=month,
        )

    def s1_gmt_params(self) -> dict:
        category = sample_weighted(
            self.category["normal"],
            self.category["normal_w"],
            self.rng,
        )

        gmt = sample_weighted(
            self.gmt["anomaly"],
            self.gmt["anomaly_w"],
            self.rng,
        )

        month = sample_month(self.rng)

        return self._params(
            scenario="s1_gmt",
            label=1,
            category=category,
            gmt=gmt,
            month=month,
        )

    def s1_mix_params(self) -> dict:
        category = sample_weighted(
            self.category["anomaly"],
            self.category["anomaly_w"],
            self.rng,
        )

        gmt = sample_weighted(
            self.gmt["anomaly"],
            self.gmt["anomaly_w"],
            self.rng,
        )

        month = sample_month(self.rng)

        return self._params(
            scenario="s1_mix",
            label=1,
            category=category,
            gmt=gmt,
            month=month,
        )

    def _params(
        self,
        scenario: str,
        label: int,
        category: str,
        gmt: float,
        month: int,
    ) -> dict:
        return {
            "label": label,
            "scenario": scenario,
            "category": str(category),
            "gmt": float(gmt),
            "year": int(YEAR),
            "month": int(month),
            "provenance_semantics": "standard_nested_sum_flattened",
        }


# ============================================================
# RENDERING
# ============================================================

def _sql_quote_escape(s: str) -> str:
    return str(s).replace("'", "''")


def _format_gmt(gmt: float) -> str:
    # DuckDB aceita -6.0, mas preferimos -6 quando é inteiro.
    if float(gmt).is_integer():
        return str(int(gmt))

    return str(float(gmt))


def _prepare_sql_params(params: dict) -> dict:
    p = dict(params)

    p["category"] = _sql_quote_escape(p["category"])
    p["gmt"] = _format_gmt(float(p["gmt"]))
    p["year"] = int(p["year"])
    p["month"] = int(p["month"])

    return p


def render_sql(params: dict) -> str:
    return SQL_TEMPLATE.format(**_prepare_sql_params(params))


def render_count(params: dict) -> str:
    return COUNT_TEMPLATE.format(**_prepare_sql_params(params))


def write_metadata(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)


# ============================================================
# WRITER
# ============================================================

def write_queries(
    out_dir: Path,
    prefix: str,
    n: int,
    sampler_fn: Callable[[], dict],
    con: duckdb.DuckDBPyConnection,
) -> list[dict]:

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []
    i = 0
    attempts = 0
    max_attempts = max(100_000, n * 50)

    while i < n:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Demasiadas tentativas sem conseguir gerar queries suficientes. "
                f"prefix={prefix}, generated={i}, requested={n}, attempts={attempts}"
            )

        params = sampler_fn()

        try:
            n_rows = con.execute(
                render_count(params)
            ).fetchone()[0]
        except Exception as exc:
            print(f"[ERROR] count failed for {params}: {exc}")
            continue

        if n_rows == 0:
            continue

        filename = f"{prefix}_{i:05d}.sql"

        (out_dir / filename).write_text(
            render_sql(params).strip() + "\n",
            encoding="utf-8",
        )

        rows.append(
            {
                "file": filename,
                **params,
                "n_rows": int(n_rows),
            }
        )

        i += 1

        if i % 1000 == 0:
            print(f"  [GEN] {prefix}: {i}/{n}")

    write_metadata(
        out_dir / "metadata.csv",
        rows,
    )

    print(f"  [OK] {prefix}: {len(rows)} queries, attempts={attempts}")

    return rows


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_overlap_for_param(
    norm: pd.DataFrame,
    anom: pd.DataFrame,
    scenario: str,
    param: str,
    should_be_disjoint: bool,
) -> None:
    g = anom[anom["scenario"] == scenario]

    if g.empty:
        return

    if param == "category":
        norm_values = set(norm["category"].astype(str))
        anom_values = set(g["category"].astype(str))
    elif param == "gmt":
        norm_values = set(norm["gmt"].astype(float))
        anom_values = set(g["gmt"].astype(float))
    elif param == "month":
        norm_values = set(norm["month"].astype(int))
        anom_values = set(g["month"].astype(int))
    else:
        raise ValueError(param)

    overlap = norm_values & anom_values
    ok = len(overlap) == 0 if should_be_disjoint else True

    print(f"  {param:<10} overlap={len(overlap):<5} {'OK' if ok else 'ERROR'}")

    if not ok:
        print(f"      overlap sample={sorted(overlap)[:20]}")


def print_diagnostics(
    normal_rows: list[dict],
    anom_rows: list[dict],
) -> None:
    norm = pd.DataFrame(normal_rows)
    anom = pd.DataFrame(anom_rows)

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    print(f"\n[NORMAL] n={len(norm)}")
    print(f"  categories unique={norm['category'].nunique()}")
    print(f"  categories={sorted(norm['category'].unique().tolist())}")
    print(f"  gmt unique={norm['gmt'].nunique()}")
    print(f"  gmt={sorted(norm['gmt'].unique().tolist())}")
    print(f"  year unique={norm['year'].nunique()} values={sorted(norm['year'].unique().tolist())}")
    print(f"  month unique={norm['month'].nunique()} values={sorted(norm['month'].unique().tolist())}")
    print(
        f"  n_rows min={norm['n_rows'].min()} "
        f"median={int(norm['n_rows'].median())} "
        f"max={norm['n_rows'].max()}"
    )

    print(f"\n[ANOMALY] n={len(anom)}")
    print(f"  scenarios={anom['scenario'].value_counts().to_dict()}")

    for scenario in sorted(anom["scenario"].unique()):
        g = anom[anom["scenario"] == scenario]

        print(f"\n[{scenario}] n={len(g)}")
        print(f"  categories unique={g['category'].nunique()}")
        print(f"  categories={sorted(g['category'].unique().tolist())}")
        print(f"  gmt unique={g['gmt'].nunique()}")
        print(f"  gmt={sorted(g['gmt'].unique().tolist())}")
        print(f"  month unique={g['month'].nunique()} values={sorted(g['month'].unique().tolist())}")
        print(
            f"  n_rows min={g['n_rows'].min()} "
            f"median={int(g['n_rows'].median())} "
            f"max={g['n_rows'].max()}"
        )

        if scenario == "s1_category":
            print_overlap_for_param(norm, anom, scenario, "category", True)
            print_overlap_for_param(norm, anom, scenario, "gmt", False)
            print_overlap_for_param(norm, anom, scenario, "month", False)

        elif scenario == "s1_gmt":
            print_overlap_for_param(norm, anom, scenario, "category", False)
            print_overlap_for_param(norm, anom, scenario, "gmt", True)
            print_overlap_for_param(norm, anom, scenario, "month", False)

        elif scenario == "s1_mix":
            print_overlap_for_param(norm, anom, scenario, "category", True)
            print_overlap_for_param(norm, anom, scenario, "gmt", True)
            print_overlap_for_param(norm, anom, scenario, "month", False)

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q60 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q60.")
    p.add_argument("--schema", default=DB_SCHEMA)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-per-type", type=int, default=N_ANOMALY_PER_TYPE)
    p.add_argument("--n-normal", type=int, default=N_NORMAL)
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
    for name in ['q60_normal', 'q60_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, EXTERNAL_BASE_DIR, N_ANOMALY_PER_TYPE, N_NORMAL, OUT_ANOM, OUT_NORMAL, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_ANOMALY_PER_TYPE = args.n_per_type
    N_NORMAL = args.n_normal
    OUT_NORMAL = args.output_dir / "q60_normal"
    OUT_ANOM = args.output_dir / "q60_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(
        f"[CONFIG] {QUERY.upper()}  "
        f"K={K}  "
        f"N_NORMAL={N_NORMAL}  "
        f"N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}"
    )
    print("[CONFIG] replacement for Q75")
    print("[CONFIG] anomaly dimensions:")
    print("  - category")
    print("  - ca_gmt_offset")
    print("[CONFIG] contextual normal parameters:")
    print(f"  - year fixed = {YEAR}")
    print(f"  - months normal = {MONTHS}")
    print("[CONFIG] provenance semantics:")
    print("  - standard why-provenance")
    print("  - per-channel witnesses are concatenated in final aggregation")
    print("  - no product between aggregated provenance sets")
    print(f"[CONFIG] generate_mix={GENERATE_MIX}")
    print(f"[CONFIG] output base={EXTERNAL_BASE_DIR}")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET search_path = ?", [DB_SCHEMA])

    rng = np.random.default_rng(SEED)

    sampler = QuerySampler(
        con,
        rng,
    )

    print("\n[GEN] normal")
    normal_rows = write_queries(
        OUT_NORMAL,
        "normal",
        N_NORMAL,
        sampler.normal_params,
        con,
    )

    anom_rows: list[dict] = []

    print("\n[GEN] s1_category")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_category",
            N_ANOMALY_PER_TYPE,
            sampler.s1_category_params,
            con,
        )
    )

    print("\n[GEN] s1_gmt")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_gmt",
            N_ANOMALY_PER_TYPE,
            sampler.s1_gmt_params,
            con,
        )
    )

    if GENERATE_MIX:
        print("\n[GEN] s1_mix")
        anom_rows.extend(
            write_queries(
                OUT_ANOM,
                "s1_mix",
                N_ANOMALY_PER_TYPE,
                sampler.s1_mix_params,
                con,
            )
        )

    write_metadata(
        OUT_ANOM / "metadata.csv",
        anom_rows,
    )

    con.close()

    print_diagnostics(
        normal_rows,
        anom_rows,
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
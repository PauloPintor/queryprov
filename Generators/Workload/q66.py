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

N_NORMAL = 65_000

# Q66 só tem uma dimensão anómala: carrier.
# Por isso pode ter mais anomalias desta classe.
N_ANOMALY_PER_TYPE = 12_500

GENERATE_MIX = False

# K=0.5 -> sampling mais concentrado nos carriers frequentes
# K=1.0 -> sampling empírico
# K>1   -> sampling mais uniforme
K = 1.0

YEAR = 2002

TIME_LO = 1
TIME_HI = 57_597
TIME_WINDOW = 28_800

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.


NETONE_OPTIONS = [
    "ws_net_paid",
    "ws_net_paid_inc_tax",
    "ws_net_paid_inc_ship",
    "ws_net_paid_inc_ship_tax",
    "ws_net_profit",
]

NETTWO_OPTIONS = [
    "cs_net_paid",
    "cs_net_paid_inc_tax",
    "cs_net_paid_inc_ship",
    "cs_net_paid_inc_ship_tax",
    "cs_net_profit",
]

SALESONE_OPTIONS = [
    "ws_sales_price",
    "ws_ext_sales_price",
    "ws_ext_list_price",
]

SALESTWO_OPTIONS = [
    "cs_sales_price",
    "cs_ext_sales_price",
    "cs_ext_list_price",
]


# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
select
        w_warehouse_name
    ,w_warehouse_sq_ft
    ,w_city
    ,w_county
    ,w_state
    ,w_country
    ,ship_carriers
    ,year
    ,sum(jan_sales) as jan_sales
    ,sum(feb_sales) as feb_sales
    ,sum(mar_sales) as mar_sales
    ,sum(apr_sales) as apr_sales
    ,sum(may_sales) as may_sales
    ,sum(jun_sales) as jun_sales
    ,sum(jul_sales) as jul_sales
    ,sum(aug_sales) as aug_sales
    ,sum(sep_sales) as sep_sales
    ,sum(oct_sales) as oct_sales
    ,sum(nov_sales) as nov_sales
    ,sum(dec_sales) as dec_sales
    ,sum(jan_sales/w_warehouse_sq_ft) as jan_sales_per_sq_foot
    ,sum(feb_sales/w_warehouse_sq_ft) as feb_sales_per_sq_foot
    ,sum(mar_sales/w_warehouse_sq_ft) as mar_sales_per_sq_foot
    ,sum(apr_sales/w_warehouse_sq_ft) as apr_sales_per_sq_foot
    ,sum(may_sales/w_warehouse_sq_ft) as may_sales_per_sq_foot
    ,sum(jun_sales/w_warehouse_sq_ft) as jun_sales_per_sq_foot
    ,sum(jul_sales/w_warehouse_sq_ft) as jul_sales_per_sq_foot
    ,sum(aug_sales/w_warehouse_sq_ft) as aug_sales_per_sq_foot
    ,sum(sep_sales/w_warehouse_sq_ft) as sep_sales_per_sq_foot
    ,sum(oct_sales/w_warehouse_sq_ft) as oct_sales_per_sq_foot
    ,sum(nov_sales/w_warehouse_sq_ft) as nov_sales_per_sq_foot
    ,sum(dec_sales/w_warehouse_sq_ft) as dec_sales_per_sq_foot
    ,sum(jan_net) as jan_net
    ,sum(feb_net) as feb_net
    ,sum(mar_net) as mar_net
    ,sum(apr_net) as apr_net
    ,sum(may_net) as may_net
    ,sum(jun_net) as jun_net
    ,sum(jul_net) as jul_net
    ,sum(aug_net) as aug_net
    ,sum(sep_net) as sep_net
    ,sum(oct_net) as oct_net
    ,sum(nov_net) as nov_net
    ,sum(dec_net) as dec_net
    ,list_reduce(array_agg(prov), (x, y) -> list_concat(x, y)) as prov
from (
    select
        w_warehouse_name
        ,w_warehouse_sq_ft
        ,w_city
        ,w_county
        ,w_state
        ,w_country
        ,'{smc1}' || ',' || '{smc2}' as ship_carriers
        ,d_year as year
        ,sum(case when d_moy = 1 then {salesone} * ws_quantity else 0 end) as jan_sales
        ,sum(case when d_moy = 2 then {salesone} * ws_quantity else 0 end) as feb_sales
        ,sum(case when d_moy = 3 then {salesone} * ws_quantity else 0 end) as mar_sales
        ,sum(case when d_moy = 4 then {salesone} * ws_quantity else 0 end) as apr_sales
        ,sum(case when d_moy = 5 then {salesone} * ws_quantity else 0 end) as may_sales
        ,sum(case when d_moy = 6 then {salesone} * ws_quantity else 0 end) as jun_sales
        ,sum(case when d_moy = 7 then {salesone} * ws_quantity else 0 end) as jul_sales
        ,sum(case when d_moy = 8 then {salesone} * ws_quantity else 0 end) as aug_sales
        ,sum(case when d_moy = 9 then {salesone} * ws_quantity else 0 end) as sep_sales
        ,sum(case when d_moy = 10 then {salesone} * ws_quantity else 0 end) as oct_sales
        ,sum(case when d_moy = 11 then {salesone} * ws_quantity else 0 end) as nov_sales
        ,sum(case when d_moy = 12 then {salesone} * ws_quantity else 0 end) as dec_sales
        ,sum(case when d_moy = 1 then {netone} * ws_quantity else 0 end) as jan_net
        ,sum(case when d_moy = 2 then {netone} * ws_quantity else 0 end) as feb_net
        ,sum(case when d_moy = 3 then {netone} * ws_quantity else 0 end) as mar_net
        ,sum(case when d_moy = 4 then {netone} * ws_quantity else 0 end) as apr_net
        ,sum(case when d_moy = 5 then {netone} * ws_quantity else 0 end) as may_net
        ,sum(case when d_moy = 6 then {netone} * ws_quantity else 0 end) as jun_net
        ,sum(case when d_moy = 7 then {netone} * ws_quantity else 0 end) as jul_net
        ,sum(case when d_moy = 8 then {netone} * ws_quantity else 0 end) as aug_net
        ,sum(case when d_moy = 9 then {netone} * ws_quantity else 0 end) as sep_net
        ,sum(case when d_moy = 10 then {netone} * ws_quantity else 0 end) as oct_net
        ,sum(case when d_moy = 11 then {netone} * ws_quantity else 0 end) as nov_net
        ,sum(case when d_moy = 12 then {netone} * ws_quantity else 0 end) as dec_net
        ,array_agg(
            list_value(
                web_sales.prov,
                warehouse.prov,
                date_dim.prov,
                time_dim.prov,
                ship_mode.prov
            )
        ) as prov
    from web_sales,
         warehouse,
         date_dim,
         time_dim,
         ship_mode
    where ws_warehouse_sk = w_warehouse_sk
      and ws_sold_date_sk = d_date_sk
      and ws_sold_time_sk = t_time_sk
      and ws_ship_mode_sk = sm_ship_mode_sk
      and d_year = {year}
      and t_time between {timeone} and {timeone}+{time_window}
      and sm_carrier in ('{smc1}', '{smc2}')
    group by
        w_warehouse_name,
        w_warehouse_sq_ft,
        w_city,
        w_county,
        w_state,
        w_country,
        d_year

    union all

    select
        w_warehouse_name
        ,w_warehouse_sq_ft
        ,w_city
        ,w_county
        ,w_state
        ,w_country
        ,'{smc1}' || ',' || '{smc2}' as ship_carriers
        ,d_year as year
        ,sum(case when d_moy = 1 then {salestwo} * cs_quantity else 0 end) as jan_sales
        ,sum(case when d_moy = 2 then {salestwo} * cs_quantity else 0 end) as feb_sales
        ,sum(case when d_moy = 3 then {salestwo} * cs_quantity else 0 end) as mar_sales
        ,sum(case when d_moy = 4 then {salestwo} * cs_quantity else 0 end) as apr_sales
        ,sum(case when d_moy = 5 then {salestwo} * cs_quantity else 0 end) as may_sales
        ,sum(case when d_moy = 6 then {salestwo} * cs_quantity else 0 end) as jun_sales
        ,sum(case when d_moy = 7 then {salestwo} * cs_quantity else 0 end) as jul_sales
        ,sum(case when d_moy = 8 then {salestwo} * cs_quantity else 0 end) as aug_sales
        ,sum(case when d_moy = 9 then {salestwo} * cs_quantity else 0 end) as sep_sales
        ,sum(case when d_moy = 10 then {salestwo} * cs_quantity else 0 end) as oct_sales
        ,sum(case when d_moy = 11 then {salestwo} * cs_quantity else 0 end) as nov_sales
        ,sum(case when d_moy = 12 then {salestwo} * cs_quantity else 0 end) as dec_sales
        ,sum(case when d_moy = 1 then {nettwo} * cs_quantity else 0 end) as jan_net
        ,sum(case when d_moy = 2 then {nettwo} * cs_quantity else 0 end) as feb_net
        ,sum(case when d_moy = 3 then {nettwo} * cs_quantity else 0 end) as mar_net
        ,sum(case when d_moy = 4 then {nettwo} * cs_quantity else 0 end) as apr_net
        ,sum(case when d_moy = 5 then {nettwo} * cs_quantity else 0 end) as may_net
        ,sum(case when d_moy = 6 then {nettwo} * cs_quantity else 0 end) as jun_net
        ,sum(case when d_moy = 7 then {nettwo} * cs_quantity else 0 end) as jul_net
        ,sum(case when d_moy = 8 then {nettwo} * cs_quantity else 0 end) as aug_net
        ,sum(case when d_moy = 9 then {nettwo} * cs_quantity else 0 end) as sep_net
        ,sum(case when d_moy = 10 then {nettwo} * cs_quantity else 0 end) as oct_net
        ,sum(case when d_moy = 11 then {nettwo} * cs_quantity else 0 end) as nov_net
        ,sum(case when d_moy = 12 then {nettwo} * cs_quantity else 0 end) as dec_net
        ,array_agg(
            list_value(
                catalog_sales.prov,
                warehouse.prov,
                date_dim.prov,
                time_dim.prov,
                ship_mode.prov
            )
        ) as prov
    from catalog_sales,
         warehouse,
         date_dim,
         time_dim,
         ship_mode
    where cs_warehouse_sk = w_warehouse_sk
      and cs_sold_date_sk = d_date_sk
      and cs_sold_time_sk = t_time_sk
      and cs_ship_mode_sk = sm_ship_mode_sk
      and d_year = {year}
      and t_time between {timeone} and {timeone}+{time_window}
      and sm_carrier in ('{smc1}', '{smc2}')
    group by
        w_warehouse_name,
        w_warehouse_sq_ft,
        w_city,
        w_county,
        w_state,
        w_country,
        d_year
) x
group by
    w_warehouse_name,
    w_warehouse_sq_ft,
    w_city,
    w_county,
    w_state,
    w_country,
    ship_carriers,
    year
order by w_warehouse_name
limit 100
"""


COUNT_TEMPLATE = """
select count(*) as n_rows
from (
    select
        w_warehouse_name,
        w_warehouse_sq_ft,
        w_city,
        w_county,
        w_state,
        w_country,
        ship_carriers,
        year
    from (
        select
            w_warehouse_name,
            w_warehouse_sq_ft,
            w_city,
            w_county,
            w_state,
            w_country,
            '{smc1}' || ',' || '{smc2}' as ship_carriers,
            d_year as year
        from web_sales,
             warehouse,
             date_dim,
             time_dim,
             ship_mode
        where ws_warehouse_sk = w_warehouse_sk
          and ws_sold_date_sk = d_date_sk
          and ws_sold_time_sk = t_time_sk
          and ws_ship_mode_sk = sm_ship_mode_sk
          and d_year = {year}
          and t_time between {timeone} and {timeone}+{time_window}
          and sm_carrier in ('{smc1}', '{smc2}')
        group by
            w_warehouse_name,
            w_warehouse_sq_ft,
            w_city,
            w_county,
            w_state,
            w_country,
            d_year

        union all

        select
            w_warehouse_name,
            w_warehouse_sq_ft,
            w_city,
            w_county,
            w_state,
            w_country,
            '{smc1}' || ',' || '{smc2}' as ship_carriers,
            d_year as year
        from catalog_sales,
             warehouse,
             date_dim,
             time_dim,
             ship_mode
        where cs_warehouse_sk = w_warehouse_sk
          and cs_sold_date_sk = d_date_sk
          and cs_sold_time_sk = t_time_sk
          and cs_ship_mode_sk = sm_ship_mode_sk
          and d_year = {year}
          and t_time between {timeone} and {timeone}+{time_window}
          and sm_carrier in ('{smc1}', '{smc2}')
        group by
            w_warehouse_name,
            w_warehouse_sq_ft,
            w_city,
            w_county,
            w_state,
            w_country,
            d_year
    ) x
    group by
        w_warehouse_name,
        w_warehouse_sq_ft,
        w_city,
        w_county,
        w_state,
        w_country,
        ship_carriers,
        year
) t
"""


# ============================================================
# HELPERS
# ============================================================

def sql_escape(s: str) -> str:
    return str(s).replace("'", "''")


def _weights_from_counts(counts: np.ndarray, k: float) -> np.ndarray:
    alpha = 1.0 / max(k, 0.05)
    w = counts.astype(float) ** alpha
    w = w / w.sum()
    return w


def sample_weighted_without_replacement(
    pool: list[str],
    weights: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> list[str]:
    if len(pool) < n:
        raise ValueError(f"Pool demasiado pequeno: {len(pool)} < {n}")

    idx = rng.choice(
        len(pool),
        size=n,
        replace=False,
        p=weights,
    )

    return [pool[int(i)] for i in idx]


# ============================================================
# CARRIER TIERS
# ============================================================

def fit_carrier_tiers(
    con: duckdb.DuckDBPyConnection,
    k: float,
    rng: np.random.Generator,
) -> dict:
    """
    Q66 uses ship carriers as the only anomalous categorical dimension.

    The carrier frequencies are estimated in the same fixed year used by
    the generated queries, so the pools reflect the executable workload
    context rather than the full database.
    """

    df = con.execute(f"""
        select
            sm.sm_carrier as carrier,
            count(*) as n_sales
        from (
            select
                ws.ws_ship_mode_sk as ship_mode_sk
            from web_sales ws
            join date_dim d
              on ws.ws_sold_date_sk = d.d_date_sk
            where ws.ws_ship_mode_sk is not null
              and d.d_year = {YEAR}

            union all

            select
                cs.cs_ship_mode_sk as ship_mode_sk
            from catalog_sales cs
            join date_dim d
              on cs.cs_sold_date_sk = d.d_date_sk
            where cs.cs_ship_mode_sk is not null
              and d.d_year = {YEAR}
        ) x
        join ship_mode sm
          on x.ship_mode_sk = sm.sm_ship_mode_sk
        where sm.sm_carrier is not null
        group by sm.sm_carrier
        order by n_sales desc,
                 carrier
    """).df()

    if len(df) < 4:
        raise ValueError(
            f"Poucos carriers distintos: {len(df)}. "
            "A Q66 precisa de pelo menos 2 carriers normais e 2 anómalos."
        )

    carriers = df["carrier"].astype(str).tolist()
    counts = df["n_sales"].astype(float).to_numpy()
    n = len(carriers)

    # Como cada query usa dois carriers, ambos os pools precisam de >= 2.
    normal_size = n // 2
    normal_size = max(2, min(normal_size, n - 2))

    base_probs = counts / counts.sum()

    normal_idx = rng.choice(
        np.arange(n),
        size=normal_size,
        replace=False,
        p=base_probs,
    )

    normal_idx = sorted(int(i) for i in normal_idx)
    normal_set = set(normal_idx)

    anomaly_idx = [i for i in range(n) if i not in normal_set]

    normal_carriers = [carriers[i] for i in normal_idx]
    anomaly_carriers = [carriers[i] for i in anomaly_idx]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    normal_w = _weights_from_counts(normal_counts, k)
    anomaly_w = _weights_from_counts(anomaly_counts, k)

    alpha = 1.0 / max(k, 0.05)

    print("\n[CARRIER]")
    print(f"  year={YEAR}")
    print(f"  n_carriers={n}  K={k}  alpha={alpha:.3f}")
    print(f"  normal n={len(normal_carriers)}  anomaly n={len(anomaly_carriers)}")
    print(f"  normal={normal_carriers}")
    print(f"  anomaly={anomaly_carriers}")

    return {
        "normal": normal_carriers,
        "normal_w": normal_w,
        "anomaly": anomaly_carriers,
        "anomaly_w": anomaly_w,
        "df": df,
    }


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
        self.carrier = fit_carrier_tiers(con, K, rng)

    def _common_params(self) -> dict:
        return {
            "year": YEAR,
            "timeone": int(self.rng.integers(TIME_LO, TIME_HI + 1)),
            "time_window": TIME_WINDOW,
            "netone": str(self.rng.choice(NETONE_OPTIONS)),
            "nettwo": str(self.rng.choice(NETTWO_OPTIONS)),
            "salesone": str(self.rng.choice(SALESONE_OPTIONS)),
            "salestwo": str(self.rng.choice(SALESTWO_OPTIONS)),
        }

    def normal_params(self) -> dict:
        carriers = sample_weighted_without_replacement(
            self.carrier["normal"],
            self.carrier["normal_w"],
            self.rng,
            2,
        )

        return self._build(
            scenario="normal",
            label=0,
            carriers=carriers,
        )

    def s1_carrier_params(self) -> dict:
        carriers = sample_weighted_without_replacement(
            self.carrier["anomaly"],
            self.carrier["anomaly_w"],
            self.rng,
            2,
        )

        return self._build(
            scenario="s1_carrier",
            label=1,
            carriers=carriers,
        )

    def _build(
        self,
        scenario: str,
        label: int,
        carriers: list[str],
    ) -> dict:
        p = self._common_params()

        return {
            "label": label,
            "scenario": scenario,
            "smc1": carriers[0],
            "smc2": carriers[1],
            **p,
        }


# ============================================================
# RENDERING
# ============================================================

def _prepare_sql_params(params: dict) -> dict:
    p = dict(params)

    p["smc1"] = sql_escape(p["smc1"])
    p["smc2"] = sql_escape(p["smc2"])

    return p


def render_sql(params: dict) -> str:
    return SQL_TEMPLATE.format(**_prepare_sql_params(params))


def render_count(params: dict) -> str:
    return COUNT_TEMPLATE.format(**_prepare_sql_params(params))


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


# ============================================================
# WRITER
# ============================================================

def write_queries(
    out_dir: Path,
    prefix: str,
    n: int,
    sampler_fn,
    con: duckdb.DuckDBPyConnection,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    i = 0
    attempts = 0

    while i < n:
        attempts += 1

        params = sampler_fn()

        try:
            n_rows = con.execute(render_count(params)).fetchone()[0]
        except Exception as e:
            print(f"[ERROR] count failed for {params}: {e}")
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
                "n_rows": n_rows,
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

def carrier_set(df: pd.DataFrame) -> set[str]:
    return set(df["smc1"].astype(str)) | set(df["smc2"].astype(str))


def print_diagnostics(
    normal_rows: list[dict],
    anom_rows: list[dict],
) -> None:
    norm = pd.DataFrame(normal_rows)
    anom = pd.DataFrame(anom_rows)

    norm_carriers = carrier_set(norm)
    anom_carriers = carrier_set(anom)

    overlap = norm_carriers & anom_carriers

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    print(f"\n[NORMAL] n={len(norm)}")
    print(f"  carriers unique={len(norm_carriers)}")
    print(f"  carriers={sorted(norm_carriers)}")
    print(f"  timeone min={norm['timeone'].min()} max={norm['timeone'].max()}")
    print(f"  year unique={sorted(norm['year'].unique().tolist())}")
    print(
        f"  n_rows min={norm['n_rows'].min()} "
        f"median={int(norm['n_rows'].median())} "
        f"max={norm['n_rows'].max()}"
    )

    print(f"\n[ANOMALY] n={len(anom)}")
    print(f"  scenarios={anom['scenario'].value_counts().to_dict()}")
    print(f"  carriers unique={len(anom_carriers)}")
    print(f"  carriers={sorted(anom_carriers)}")
    print(f"  timeone min={anom['timeone'].min()} max={anom['timeone'].max()}")
    print(f"  year unique={sorted(anom['year'].unique().tolist())}")
    print(
        f"  n_rows min={anom['n_rows'].min()} "
        f"median={int(anom['n_rows'].median())} "
        f"max={anom['n_rows'].max()}"
    )

    print("\n[OVERLAP]")
    print(f"  carrier overlap={len(overlap)} {'OK' if len(overlap) == 0 else 'ERROR'}")

    if overlap:
        print(f"  overlapping carriers={sorted(overlap)}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q66 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q66.")
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
    for name in ['q66_normal', 'q66_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q66_normal"
    OUT_ANOM = args.output_dir / "q66_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(
        f"[CONFIG] Q66  "
        f"K={K}  "
        f"N_NORMAL={N_NORMAL}  "
        f"N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}"
    )
    print(f"[CONFIG] fixed year={YEAR}")
    print(f"[CONFIG] time uniform=[{TIME_LO}, {TIME_HI}] window={TIME_WINDOW}")
    print("[CONFIG] anomaly dimensions:")
    print("  - ship carrier only")
    print("[CONFIG] time and metric columns are normal workload variation")
    print(f"[CONFIG] generate_mix={GENERATE_MIX}")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET search_path = ?", [DB_SCHEMA])

    rng = np.random.default_rng(SEED)
    sampler = QuerySampler(con, rng)

    print("\n[GEN] normal")
    normal_rows = write_queries(
        OUT_NORMAL,
        "normal",
        N_NORMAL,
        sampler.normal_params,
        con,
    )

    anom_rows: list[dict] = []

    print("\n[GEN] s1_carrier")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_carrier",
            N_ANOMALY_PER_TYPE,
            sampler.s1_carrier_params,
            con,
        )
    )

    write_metadata(
        OUT_ANOM / "metadata.csv",
        anom_rows,
    )

    con.close()

    print_diagnostics(normal_rows, anom_rows)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
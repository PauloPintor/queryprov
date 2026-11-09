#!/usr/bin/env python3
"""Portable Q5 normal and S1 workload generator.

Run with --help for paths and options. Requires duckdb, numpy and pandas.
The database is opened read-only. Original SQL and sampling are preserved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DB_SCHEMA = "tpcds"

SEED = 42

N_NORMAL = 65_000

# Q5 só tem anomalia temporal.
N_ANOMALY = 1_050

K = 1.0

DATE_WINDOW_DAYS = 14

NORMAL_CONTEXT_MIN = date(2002, 1, 1)
NORMAL_CONTEXT_MAX = date(2002, 6, 30)

ANOM_CONTEXT_MIN = date(2001, 7, 1)
ANOM_CONTEXT_MAX = date(2001, 12, 31)

RECENCY_EXP_SCALE = 0.25



# ============================================================
# SQL TEMPLATE
# ============================================================

SQL_TEMPLATE = """
with ssr as
(
    select s_store_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, store.prov)
               )
           ) as prov
    from
    (
        select ss_store_sk as store_sk,
               ss_sold_date_sk as date_sk,
               ss_ext_sales_price as sales_price,
               ss_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(store_sales.prov) as src_prov
        from store_sales

        union all

        select sr_store_sk as store_sk,
               sr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               sr_return_amt as return_amt,
               sr_net_loss as net_loss,
               list_value(store_returns.prov) as src_prov
        from store_returns
    ) salesreturns,
      date_dim,
      store
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and store_sk = s_store_sk
    group by s_store_id
),
csr as
(
    select cp_catalog_page_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, catalog_page.prov)
               )
           ) as prov
    from
    (
        select cs_catalog_page_sk as page_sk,
               cs_sold_date_sk as date_sk,
               cs_ext_sales_price as sales_price,
               cs_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(catalog_sales.prov) as src_prov
        from catalog_sales

        union all

        select cr_catalog_page_sk as page_sk,
               cr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               cr_return_amount as return_amt,
               cr_net_loss as net_loss,
               list_value(catalog_returns.prov) as src_prov
        from catalog_returns
    ) salesreturns,
      date_dim,
      catalog_page
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and page_sk = cp_catalog_page_sk
    group by cp_catalog_page_id
),
wsr as
(
    select web_site_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss,
           array_agg(
               list_concat(
                   src_prov,
                   list_value(date_dim.prov, web_site.prov)
               )
           ) as prov
    from
    (
        select ws_web_site_sk as wsr_web_site_sk,
               ws_sold_date_sk as date_sk,
               ws_ext_sales_price as sales_price,
               ws_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss,
               list_value(web_sales.prov) as src_prov
        from web_sales

        union all

        select ws_web_site_sk as wsr_web_site_sk,
               wr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               wr_return_amt as return_amt,
               wr_net_loss as net_loss,
               case
                   when web_sales.prov is null then list_value(web_returns.prov)
                   else list_value(web_returns.prov, web_sales.prov)
               end as src_prov
        from web_returns
        left outer join web_sales
          on wr_item_sk = ws_item_sk
         and wr_order_number = ws_order_number
    ) salesreturns,
      date_dim,
      web_site
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and wsr_web_site_sk = web_site_sk
    group by web_site_id
)
select channel,
       id,
       sum(sales) as sales,
       sum(returns) as returns,
       sum(profit) as profit,
       array_agg(prov) as prov
from
(
    select 'store channel' as channel,
           'store' || s_store_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from ssr

    union all

    select 'catalog channel' as channel,
           'catalog_page' || cp_catalog_page_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from csr

    union all

    select 'web channel' as channel,
           'web_site' || web_site_id as id,
           sales,
           returns,
           profit - profit_loss as profit,
           prov
    from wsr
) x
group by rollup(channel, id)
order by channel, id
limit 100
"""


COUNT_TEMPLATE = """
with ssr as
(
    select s_store_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss
    from
    (
        select ss_store_sk as store_sk,
               ss_sold_date_sk as date_sk,
               ss_ext_sales_price as sales_price,
               ss_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss
        from store_sales

        union all

        select sr_store_sk as store_sk,
               sr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               sr_return_amt as return_amt,
               sr_net_loss as net_loss
        from store_returns
    ) salesreturns,
      date_dim,
      store
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and store_sk = s_store_sk
    group by s_store_id
),
csr as
(
    select cp_catalog_page_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss
    from
    (
        select cs_catalog_page_sk as page_sk,
               cs_sold_date_sk as date_sk,
               cs_ext_sales_price as sales_price,
               cs_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss
        from catalog_sales

        union all

        select cr_catalog_page_sk as page_sk,
               cr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               cr_return_amount as return_amt,
               cr_net_loss as net_loss
        from catalog_returns
    ) salesreturns,
      date_dim,
      catalog_page
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and page_sk = cp_catalog_page_sk
    group by cp_catalog_page_id
),
wsr as
(
    select web_site_id,
           sum(sales_price) as sales,
           sum(profit) as profit,
           sum(return_amt) as returns,
           sum(net_loss) as profit_loss
    from
    (
        select ws_web_site_sk as wsr_web_site_sk,
               ws_sold_date_sk as date_sk,
               ws_ext_sales_price as sales_price,
               ws_net_profit as profit,
               cast(0 as decimal(7,2)) as return_amt,
               cast(0 as decimal(7,2)) as net_loss
        from web_sales

        union all

        select ws_web_site_sk as wsr_web_site_sk,
               wr_returned_date_sk as date_sk,
               cast(0 as decimal(7,2)) as sales_price,
               cast(0 as decimal(7,2)) as profit,
               wr_return_amt as return_amt,
               wr_net_loss as net_loss
        from web_returns
        left outer join web_sales
          on wr_item_sk = ws_item_sk
         and wr_order_number = ws_order_number
    ) salesreturns,
      date_dim,
      web_site
    where date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
      and wsr_web_site_sk = web_site_sk
    group by web_site_id
)
select count(*)
from
(
    select channel,
           id,
           sum(sales) as sales,
           sum(returns) as returns,
           sum(profit) as profit
    from
    (
        select 'store channel' as channel,
               'store' || s_store_id as id,
               sales,
               returns,
               profit - profit_loss as profit
        from ssr

        union all

        select 'catalog channel' as channel,
               'catalog_page' || cp_catalog_page_id as id,
               sales,
               returns,
               profit - profit_loss as profit
        from csr

        union all

        select 'web channel' as channel,
               'web_site' || web_site_id as id,
               sales,
               returns,
               profit - profit_loss as profit
        from wsr
    ) x
    group by rollup(channel, id)
) t
"""


# ============================================================
# DATE HELPERS
# ============================================================

def valid_sdate_range(
    context_min: date,
    context_max: date,
    window_days: int,
) -> tuple[date, date]:
    sdate_min = context_min
    sdate_max = context_max - timedelta(days=window_days)

    if sdate_max < sdate_min:
        raise ValueError(
            f"Temporal context [{context_min}, {context_max}] "
            f"is too small for a {window_days}-day window."
        )

    return sdate_min, sdate_max


def date_to_idx(d: date, lo: date) -> int:
    return (d - lo).days


def idx_to_date(i: int, lo: date) -> date:
    return lo + timedelta(days=int(i))


def sample_recent_reflected_exponential(
    lo: int,
    hi: int,
    rng: np.random.Generator,
    scale: float = RECENCY_EXP_SCALE,
) -> int:
    width = hi - lo

    if width <= 0:
        return hi

    for _ in range(100_000):
        u = rng.exponential(scale=scale)
        if 0 <= u <= 1:
            return int(round(hi - u * width))

    return hi


def sample_uniform_date_idx(
    lo: int,
    hi: int,
    rng: np.random.Generator,
) -> int:
    return int(rng.integers(lo, hi + 1))


# ============================================================
# SAMPLER
# ============================================================

class QuerySampler:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

        self.normal_sdate_min, self.normal_sdate_max = valid_sdate_range(
            NORMAL_CONTEXT_MIN,
            NORMAL_CONTEXT_MAX,
            DATE_WINDOW_DAYS,
        )

        self.anom_sdate_min, self.anom_sdate_max = valid_sdate_range(
            ANOM_CONTEXT_MIN,
            ANOM_CONTEXT_MAX,
            DATE_WINDOW_DAYS,
        )

        self.normal_lo = 0
        self.normal_hi = date_to_idx(self.normal_sdate_max, self.normal_sdate_min)

        self.anom_lo = 0
        self.anom_hi = date_to_idx(self.anom_sdate_max, self.anom_sdate_min)

        print("\n[DATE CONTEXT]")
        print(f"  window_days={DATE_WINDOW_DAYS}")
        print(f"  normal context : [{NORMAL_CONTEXT_MIN}, {NORMAL_CONTEXT_MAX}]")
        print(f"  anomaly context: [{ANOM_CONTEXT_MIN}, {ANOM_CONTEXT_MAX}]")
        print(f"  normal SDATE   : [{self.normal_sdate_min}, {self.normal_sdate_max}]")
        print(f"  anomaly SDATE  : [{self.anom_sdate_min}, {self.anom_sdate_max}]")

    def normal_params(self) -> dict:
        d_idx = sample_recent_reflected_exponential(
            self.normal_lo,
            self.normal_hi,
            self.rng,
        )

        d = idx_to_date(d_idx, self.normal_sdate_min)

        return self._params(
            scenario="normal",
            label=0,
            sdate=d,
            date_idx=d_idx,
        )

    def s1_date_params(self) -> dict:
        d_idx = sample_uniform_date_idx(
            self.anom_lo,
            self.anom_hi,
            self.rng,
        )

        d = idx_to_date(d_idx, self.anom_sdate_min)

        return self._params(
            scenario="s1_date",
            label=1,
            sdate=d,
            date_idx=d_idx,
        )

    def _params(
        self,
        scenario: str,
        label: int,
        sdate: date,
        date_idx: int,
    ) -> dict:
        return {
            "label": label,
            "scenario": scenario,
            "sdate": sdate.isoformat(),
            "window_days": DATE_WINDOW_DAYS,
            "date_idx": date_idx,
            "window_start": sdate.isoformat(),
            "window_end": (sdate + timedelta(days=DATE_WINDOW_DAYS)).isoformat(),
        }


# ============================================================
# RENDERING
# ============================================================

def _render_sql(params: dict) -> str:
    return SQL_TEMPLATE.format(**params)


def _render_count(params: dict) -> str:
    return COUNT_TEMPLATE.format(**params)


def _write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
            n_rows = con.execute(_render_count(params)).fetchone()[0]
        except Exception as e:
            print(f"[ERROR] count failed for {params}: {e}")
            continue

        if n_rows == 0:
            continue

        filename = f"{prefix}_{i:05d}.sql"

        (out_dir / filename).write_text(
            _render_sql(params).strip() + "\n",
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

    _write_metadata(out_dir / "metadata.csv", rows)

    print(f"  [OK] {prefix}: {len(rows)} queries, attempts={attempts}")

    return rows


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_diagnostics(normal_rows: list[dict], anom_rows: list[dict]) -> None:
    norm = pd.DataFrame(normal_rows)
    anom = pd.DataFrame(anom_rows)

    norm_dates = set(norm["sdate"])
    anom_dates = set(anom["sdate"])

    norm_windows = set(zip(norm["window_start"], norm["window_end"]))
    anom_windows = set(zip(anom["window_start"], anom["window_end"]))

    date_overlap = norm_dates & anom_dates
    window_overlap = norm_windows & anom_windows

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    print(f"\n[NORMAL] n={len(norm)}")
    print(f"  unique sdate={len(norm_dates)}")
    print(f"  sdate min={norm['sdate'].min()} max={norm['sdate'].max()}")
    print(f"  window_start min={norm['window_start'].min()} max={norm['window_start'].max()}")
    print(f"  window_end   min={norm['window_end'].min()} max={norm['window_end'].max()}")
    print(
        f"  n_rows min={norm['n_rows'].min()} "
        f"median={int(norm['n_rows'].median())} "
        f"max={norm['n_rows'].max()}"
    )

    print(f"\n[ANOMALY] n={len(anom)}")
    print(f"  scenarios={anom['scenario'].value_counts().to_dict()}")
    print(f"  unique sdate={len(anom_dates)}")
    print(f"  sdate min={anom['sdate'].min()} max={anom['sdate'].max()}")
    print(f"  window_start min={anom['window_start'].min()} max={anom['window_start'].max()}")
    print(f"  window_end   min={anom['window_end'].min()} max={anom['window_end'].max()}")
    print(
        f"  n_rows min={anom['n_rows'].min()} "
        f"median={int(anom['n_rows'].median())} "
        f"max={anom['n_rows'].max()}"
    )

    print("\n[OVERLAP]")
    print(f"  date overlap={len(date_overlap)} {'OK' if len(date_overlap) == 0 else 'ERROR'}")
    print(f"  window overlap={len(window_overlap)} {'OK' if len(window_overlap) == 0 else 'ERROR'}")

    print("\n[EXPECTED]")
    print(f"  normal context must stay within : [{NORMAL_CONTEXT_MIN}, {NORMAL_CONTEXT_MAX}]")
    print(f"  anomaly context must stay within: [{ANOM_CONTEXT_MIN}, {ANOM_CONTEXT_MAX}]")

    norm_window_end_max = pd.to_datetime(norm["window_end"]).max().date()
    anom_window_end_max = pd.to_datetime(anom["window_end"]).max().date()

    print("\n[WINDOW CHECK]")
    print(
        f"  normal max window_end={norm_window_end_max} "
        f"{'OK' if norm_window_end_max <= NORMAL_CONTEXT_MAX else 'ERROR'}"
    )
    print(
        f"  anomaly max window_end={anom_window_end_max} "
        f"{'OK' if anom_window_end_max <= ANOM_CONTEXT_MAX else 'ERROR'}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Q5 normal and S1 SQL files with metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", type=Path, required=True, help="Existing DuckDB database.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent of q5_normal and q5_anomalies.")
    parser.add_argument("--schema", default=DB_SCHEMA, help="Database search path.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--n-normal", type=int, default=N_NORMAL, help="Number of normal queries.")
    parser.add_argument("--n-anomaly", type=int, default=N_ANOMALY, help="Number of S1 queries.")
    args = parser.parse_args(argv)
    args.db = args.db.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.db.is_file():
        parser.error(f"Database file does not exist: {args.db}")
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.n_normal <= 0 or args.n_anomaly <= 0:
        parser.error("--n-normal and --n-anomaly must be positive.")
    if not args.schema.strip():
        parser.error("--schema must not be empty.")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        parser.error(f"Output path is not a directory: {args.output_dir}")
    for pool in ("q5_normal", "q5_anomalies"):
        directory = args.output_dir / pool
        if directory.exists() and (
            not directory.is_dir() or next(directory.iterdir(), None) is not None
        ):
            parser.error(f"Output pool must be absent or empty: {directory}. Use a new output directory.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"[DB] {args.db}  schema={args.schema}")
    print(
        f"[CONFIG] Q5 temporal-only  "
        f"K={K}  "
        f"N_NORMAL={args.n_normal}  "
        f"N_ANOMALY={args.n_anomaly}  "
        f"SEED={args.seed}"
    )
    print("[CONFIG] normal date = reflected exponential over recent context")
    print("[CONFIG] anomaly date = uniform over older context")
    print("[CONFIG] full query window is constrained to remain inside its context")

    con = duckdb.connect(str(args.db), read_only=True)
    try:
        con.execute("SET search_path = ?", [args.schema])

        rng = np.random.default_rng(args.seed)
        sampler = QuerySampler(rng)

        print("\n[GEN] normal")
        normal_rows = write_queries(
            (args.output_dir / "q5_normal"),
            "normal",
            args.n_normal,
            sampler.normal_params,
            con,
        )

        print("\n[GEN] s1_date")
        anom_rows = write_queries(
            (args.output_dir / "q5_anomalies"),
            "s1_date",
            args.n_anomaly,
            sampler.s1_date_params,
            con,
        )

        _write_metadata((args.output_dir / "q5_anomalies") / "metadata.csv", anom_rows)

    finally:
        con.close()

    print_diagnostics(normal_rows, anom_rows)

    print("\n[DONE]")


if __name__ == "__main__":
    main()

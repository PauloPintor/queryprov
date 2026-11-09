#!/usr/bin/env python3
"""
Generate Q3 normal and S1 SQL workloads and their metadata.

Example:
    python generate_q3.py --db /path/tpcds_prov.duckdb --output-dir Queries/Q3

Requires duckdb, numpy and pandas. The database must contain the TPC-DS
tables used below.
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

N_NORMAL = 65_000
N_ANOMALY = 12_500

# Range do parâmetro MANUFACT
MANUFACT_LO = 1
MANUFACT_HI = 1000

# K=1.0 -> normal central compacto, aproximadamente [mu-sigma, mu+sigma]
# K controla a largura da região normal.
K = 1.0

# MONTH uniforme; não é dimensão anómala.
MONTH_LO = 1
MONTH_HI = 12

# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
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
limit 100
"""


COUNT_TEMPLATE = """
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
# CONTINUOUS RANGE
# ============================================================

def compute_ranges(
    lo: int,
    hi: int,
    k: float,
) -> dict:
    """
    Deriva mu, sigma e regiões normal/anómala a partir do domínio.

    mu    = (lo + hi) / 2
    sigma = (hi - lo) / 6

    normal:
        [mu - K*sigma, mu + K*sigma]

    anomaly:
        [lo, normal_lo - 1] U [normal_hi + 1, hi]
    """

    mu = (lo + hi) / 2.0
    sigma = (hi - lo) / 6.0

    lo_normal = max(lo, int(np.floor(mu - k * sigma)))
    hi_normal = min(hi, int(np.ceil(mu + k * sigma)))

    lo_anom_left = lo
    hi_anom_left = lo_normal - 1

    lo_anom_right = hi_normal + 1
    hi_anom_right = hi

    has_left = hi_anom_left >= lo_anom_left
    has_right = lo_anom_right <= hi_anom_right

    if not has_left and not has_right:
        raise ValueError(
            f"Sem região anómala para domínio [{lo}, {hi}] com K={k}."
        )

    print("\n[MANUFACT]")
    print(f"  domain=[{lo}, {hi}]")
    print(f"  K={k}")
    print(f"  mu={mu:.2f} sigma={sigma:.2f}")
    print(f"  normal=[{lo_normal}, {hi_normal}]")

    parts = []
    if has_left:
        parts.append(f"[{lo_anom_left}, {hi_anom_left}]")
    if has_right:
        parts.append(f"[{lo_anom_right}, {hi_anom_right}]")
    print(f"  anomaly={parts}")

    return {
        "lo": lo,
        "hi": hi,
        "mu": mu,
        "sigma": sigma,
        "lo_normal": lo_normal,
        "hi_normal": hi_normal,
        "lo_anom_left": lo_anom_left,
        "hi_anom_left": hi_anom_left,
        "lo_anom_right": lo_anom_right,
        "hi_anom_right": hi_anom_right,
        "has_left": has_left,
        "has_right": has_right,
    }


def sample_truncated_normal(
    mu: float,
    sigma: float,
    lo: int,
    hi: int,
    rng: np.random.Generator,
) -> int:
    for _ in range(100_000):
        v = int(round(rng.normal(mu, sigma)))

        if lo <= v <= hi:
            return v

    return int(np.clip(round(mu), lo, hi))


def sample_normal_manufact(
    ranges: dict,
    rng: np.random.Generator,
) -> int:
    """
    Normal sampling uses an effective sigma inside the normal interval.
    This keeps the normal workload concentrated inside the compact profile.
    """

    sigma_eff = max(
        (ranges["hi_normal"] - ranges["lo_normal"]) / 6.0,
        1e-6,
    )

    return sample_truncated_normal(
        ranges["mu"],
        sigma_eff,
        ranges["lo_normal"],
        ranges["hi_normal"],
        rng,
    )


def sample_anomaly_manufact(
    ranges: dict,
    rng: np.random.Generator,
) -> int:
    """
    Anomaly sampling uses the complementary tails.
    If both tails exist, left/right tails are selected with equal probability.
    """

    if ranges["has_left"] and ranges["has_right"]:
        if rng.random() < 0.5:
            lo, hi = ranges["lo_anom_left"], ranges["hi_anom_left"]
        else:
            lo, hi = ranges["lo_anom_right"], ranges["hi_anom_right"]
    elif ranges["has_left"]:
        lo, hi = ranges["lo_anom_left"], ranges["hi_anom_left"]
    elif ranges["has_right"]:
        lo, hi = ranges["lo_anom_right"], ranges["hi_anom_right"]
    else:
        raise ValueError("Sem região anómala.")

    return sample_truncated_normal(
        ranges["mu"],
        ranges["sigma"],
        lo,
        hi,
        rng,
    )


# ============================================================
# SAMPLER
# ============================================================

class QuerySampler:
    def __init__(
        self,
        rng: np.random.Generator,
    ):
        self.rng = rng
        self.ranges = compute_ranges(
            MANUFACT_LO,
            MANUFACT_HI,
            K,
        )

    def normal_params(self) -> dict:
        manufact = sample_normal_manufact(
            self.ranges,
            self.rng,
        )

        month = int(
            self.rng.integers(
                MONTH_LO,
                MONTH_HI + 1,
            )
        )

        return {
            "label": 0,
            "scenario": "normal",
            "manufact": int(manufact),
            "month": month,
        }

    def s1_manufact_params(self) -> dict:
        manufact = sample_anomaly_manufact(
            self.ranges,
            self.rng,
        )

        month = int(
            self.rng.integers(
                MONTH_LO,
                MONTH_HI + 1,
            )
        )

        return {
            "label": 1,
            "scenario": "s1_manufact",
            "manufact": int(manufact),
            "month": month,
        }


# ============================================================
# RENDERING
# ============================================================

def render_sql(params: dict) -> str:
    return SQL_TEMPLATE.format(**params)


def render_count(params: dict) -> str:
    return COUNT_TEMPLATE.format(**params)


def write_metadata(
    path: Path,
    rows: list[dict],
) -> None:
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

    max_attempts = max(100_000, n * 50)

    while i < n:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Too many attempts while generating {prefix}: "
                f"{attempts} attempts for {i}/{n} queries. "
                "The parameter space may be too small."
            )

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

def print_diagnostics(
    normal_rows: list[dict],
    anom_rows: list[dict],
) -> None:
    norm = pd.DataFrame(normal_rows)
    anom = pd.DataFrame(anom_rows)

    norm_manufact = set(norm["manufact"].astype(int))
    anom_manufact = set(anom["manufact"].astype(int))

    overlap = norm_manufact & anom_manufact

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    print(f"\n[NORMAL] n={len(norm)}")
    print(
        f"  manufact unique={len(norm_manufact)} "
        f"min={int(norm['manufact'].min())} "
        f"max={int(norm['manufact'].max())} "
        f"mean={norm['manufact'].mean():.2f} "
        f"std={norm['manufact'].std():.2f}"
    )
    print(f"  month unique={sorted(norm['month'].unique().tolist())}")
    print(
        f"  n_rows min={norm['n_rows'].min()} "
        f"median={int(norm['n_rows'].median())} "
        f"max={norm['n_rows'].max()}"
    )

    print(f"\n[ANOMALY] n={len(anom)}")
    print(f"  scenarios={anom['scenario'].value_counts().to_dict()}")
    print(
        f"  manufact unique={len(anom_manufact)} "
        f"min={int(anom['manufact'].min())} "
        f"max={int(anom['manufact'].max())} "
        f"mean={anom['manufact'].mean():.2f} "
        f"std={anom['manufact'].std():.2f}"
    )
    print(f"  month unique={sorted(anom['month'].unique().tolist())}")
    print(
        f"  n_rows min={anom['n_rows'].min()} "
        f"median={int(anom['n_rows'].median())} "
        f"max={anom['n_rows'].max()}"
    )

    print("\n[OVERLAP]")
    print(f"  manufact overlap={len(overlap)} {'OK' if len(overlap) == 0 else 'ERROR'}")

    if overlap:
        print(f"  overlapping manufact={sorted(overlap)[:20]}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Q3 normal and S1 SQL files with metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", type=Path, required=True, help="Existing DuckDB database.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent of q3_normal and q3_anomalies.")
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
    for pool in ("q3_normal", "q3_anomalies"):
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
        f"[CONFIG] Q3  "
        f"K={K}  "
        f"N_NORMAL={args.n_normal}  "
        f"N_ANOMALY={args.n_anomaly}  "
        f"SEED={args.seed}"
    )
    print("[CONFIG] anomaly dimensions:")
    print("  - manufacturer id")
    print("[CONFIG] month is sampled uniformly and is not anomalous")
    print(f"[CONFIG] output base={args.output_dir}")

    con = duckdb.connect(str(args.db), read_only=True)
    try:
        con.execute("SET search_path = ?", [args.schema])
        rng = np.random.default_rng(args.seed)
        sampler = QuerySampler(rng)

        print("\n[GEN] normal")
        normal_rows = write_queries(
            args.output_dir / "q3_normal",
            "normal",
            args.n_normal,
            sampler.normal_params,
            con,
        )

        print("\n[GEN] s1_manufact")
        anom_rows = write_queries(
            args.output_dir / "q3_anomalies",
            "s1_manufact",
            args.n_anomaly,
            sampler.s1_manufact_params,
            con,
        )
    finally:
        con.close()

    print_diagnostics(
        normal_rows,
        anom_rows,
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()

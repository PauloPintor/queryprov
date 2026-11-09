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
N_ANOMALY_PER_TYPE = 350

GENERATE_MIX = True

# K=0.5 -> sampling mais concentrado / normal mais compacto
# K=1.0 -> sampling empírico
# K>1   -> sampling mais uniforme
K = 1.0

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.

# INCOME parameter domain
INCOME_LO = 0
INCOME_HI = 70_000
INCOME_WINDOW = 50_000


# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
select
    c_customer_id as customer_id,
    coalesce(c_last_name, '') || ', ' || coalesce(c_first_name, '') as customername,
    list_value(
        customer.prov,
        customer_address.prov,
        customer_demographics.prov,
        household_demographics.prov,
        income_band.prov,
        store_returns.prov
    ) as prov
from customer,
     customer_address,
     customer_demographics,
     household_demographics,
     income_band,
     store_returns
where ca_city = '{city}'
  and c_current_addr_sk = ca_address_sk
  and ib_lower_bound >= {income}
  and ib_upper_bound <= {income} + {income_window}
  and ib_income_band_sk = hd_income_band_sk
  and cd_demo_sk = c_current_cdemo_sk
  and hd_demo_sk = c_current_hdemo_sk
  and sr_cdemo_sk = cd_demo_sk
order by
    c_customer_id
limit 100
"""


COUNT_TEMPLATE = """
select count(*) as n_rows
from (
    select
        c_customer_id
    from customer,
         customer_address,
         customer_demographics,
         household_demographics,
         income_band,
         store_returns
    where ca_city = '{city}'
      and c_current_addr_sk = ca_address_sk
      and ib_lower_bound >= {income}
      and ib_upper_bound <= {income} + {income_window}
      and ib_income_band_sk = hd_income_band_sk
      and cd_demo_sk = c_current_cdemo_sk
      and hd_demo_sk = c_current_hdemo_sk
      and sr_cdemo_sk = cd_demo_sk
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
# CITY TIERS
# ============================================================

def fit_city_tiers(
    con: duckdb.DuckDBPyConnection,
    k: float,
    rng: np.random.Generator,
) -> dict:
    """
    Q84 uses ca_city as a categorical geographical parameter.

    City frequencies are estimated over the executable query space, using
    the same joins as the query, rather than over customer_address alone.
    """

    df = con.execute("""
        select
            ca.ca_city as city,
            count(*) as n_rows
        from customer c
        join customer_address ca
          on c.c_current_addr_sk = ca.ca_address_sk
        join customer_demographics cd
          on cd.cd_demo_sk = c.c_current_cdemo_sk
        join household_demographics hd
          on hd.hd_demo_sk = c.c_current_hdemo_sk
        join income_band ib
          on ib.ib_income_band_sk = hd.hd_income_band_sk
        join store_returns sr
          on sr.sr_cdemo_sk = cd.cd_demo_sk
        where ca.ca_city is not null
        group by ca.ca_city
        order by n_rows desc,
                 city
    """).df()

    if len(df) < 6:
        raise ValueError(f"Poucas cidades distintas: {len(df)}")

    cities = df["city"].astype(str).tolist()
    counts = df["n_rows"].astype(float).to_numpy()
    n = len(cities)

    # Pool size is independent of K.
    # K controls only the concentration of the sampling weights.
    normal_size = n // 2
    normal_size = max(3, min(normal_size, n - 3))

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

    normal_cities = [cities[i] for i in normal_idx]
    anomaly_cities = [cities[i] for i in anomaly_idx]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    normal_w = _weights_from_counts(normal_counts, k)
    anomaly_w = _weights_from_counts(anomaly_counts, k)

    alpha = 1.0 / max(k, 0.05)

    print("\n[CITY]")
    print(f"  n_cities={n}  K={k}  alpha={alpha:.3f}")
    print(f"  normal n={len(normal_cities)}  anomaly n={len(anomaly_cities)}")
    print(f"  normal sample={normal_cities[:10]}")
    print(f"  anomaly sample={anomaly_cities[:10]}")

    return {
        "normal": normal_cities,
        "normal_w": normal_w,
        "anomaly": anomaly_cities,
        "anomaly_w": anomaly_w,
        "df": df,
    }


# ============================================================
# INCOME RANGES
# ============================================================

def compute_income_ranges(
    lo: int,
    hi: int,
    k: float,
) -> dict:
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
            f"Sem região anómala para INCOME no domínio [{lo}, {hi}] com K={k}."
        )

    print("\n[INCOME]")
    print(f"  domain=[{lo}, {hi}]  window={INCOME_WINDOW}")
    print(f"  K={k}")
    print(f"  mu={mu:.2f}  sigma={sigma:.2f}")
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


def sample_income_normal(
    r: dict,
    rng: np.random.Generator,
) -> int:
    sigma_eff = max((r["hi_normal"] - r["lo_normal"]) / 6.0, 1e-6)

    return sample_truncated_normal(
        r["mu"],
        sigma_eff,
        r["lo_normal"],
        r["hi_normal"],
        rng,
    )


def sample_income_anomaly(
    r: dict,
    rng: np.random.Generator,
) -> int:
    if r["has_left"] and r["has_right"]:
        if rng.random() < 0.5:
            lo, hi = r["lo_anom_left"], r["hi_anom_left"]
        else:
            lo, hi = r["lo_anom_right"], r["hi_anom_right"]
    elif r["has_left"]:
        lo, hi = r["lo_anom_left"], r["hi_anom_left"]
    elif r["has_right"]:
        lo, hi = r["lo_anom_right"], r["hi_anom_right"]
    else:
        raise ValueError("Sem região anómala para INCOME.")

    return sample_truncated_normal(
        r["mu"],
        r["sigma"],
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
        con: duckdb.DuckDBPyConnection,
        rng: np.random.Generator,
    ):
        self.rng = rng
        self.city = fit_city_tiers(con, K, rng)
        self.income_ranges = compute_income_ranges(
            INCOME_LO,
            INCOME_HI,
            K,
        )

    def normal_params(self) -> dict:
        city = sample_weighted_without_replacement(
            self.city["normal"],
            self.city["normal_w"],
            self.rng,
            1,
        )[0]

        income = sample_income_normal(
            self.income_ranges,
            self.rng,
        )

        return self._build(
            scenario="normal",
            label=0,
            city=city,
            income=income,
        )

    def s1_city_params(self) -> dict:
        city = sample_weighted_without_replacement(
            self.city["anomaly"],
            self.city["anomaly_w"],
            self.rng,
            1,
        )[0]

        income = sample_income_normal(
            self.income_ranges,
            self.rng,
        )

        return self._build(
            scenario="s1_city",
            label=1,
            city=city,
            income=income,
        )

    def s1_income_params(self) -> dict:
        city = sample_weighted_without_replacement(
            self.city["normal"],
            self.city["normal_w"],
            self.rng,
            1,
        )[0]

        income = sample_income_anomaly(
            self.income_ranges,
            self.rng,
        )

        return self._build(
            scenario="s1_income",
            label=1,
            city=city,
            income=income,
        )

    def s1_mix_params(self) -> dict:
        city = sample_weighted_without_replacement(
            self.city["anomaly"],
            self.city["anomaly_w"],
            self.rng,
            1,
        )[0]

        income = sample_income_anomaly(
            self.income_ranges,
            self.rng,
        )

        return self._build(
            scenario="s1_mix",
            label=1,
            city=city,
            income=income,
        )

    def _build(
        self,
        scenario: str,
        label: int,
        city: str,
        income: int,
    ) -> dict:
        return {
            "label": label,
            "scenario": scenario,
            "city": city,
            "income": int(income),
            "income_window": INCOME_WINDOW,
        }


# ============================================================
# RENDERING
# ============================================================

def _prepare_sql_params(params: dict) -> dict:
    p = dict(params)
    p["city"] = sql_escape(p["city"])
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

    norm_cities = set(norm["city"].astype(str))
    norm_income = set(norm["income"].astype(int))

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    print(f"\n[NORMAL] n={len(norm)}")
    print(f"  cities unique={len(norm_cities)}")
    print(f"  cities sample={sorted(norm_cities)[:20]}")
    print(f"  income unique={len(norm_income)}")
    print(f"  income min={int(norm['income'].min())} max={int(norm['income'].max())}")
    print(
        f"  n_rows min={norm['n_rows'].min()} "
        f"median={int(norm['n_rows'].median())} "
        f"max={norm['n_rows'].max()}"
    )

    print(f"\n[ANOMALY] n={len(anom)}")
    print(f"  scenarios={anom['scenario'].value_counts().to_dict()}")

    for scenario in sorted(anom["scenario"].unique()):
        g = anom[anom["scenario"] == scenario]

        g_cities = set(g["city"].astype(str))
        g_income = set(g["income"].astype(int))

        city_overlap = norm_cities & g_cities
        income_overlap = norm_income & g_income

        if scenario == "s1_city":
            city_ok = len(city_overlap) == 0
            income_ok = True
        elif scenario == "s1_income":
            city_ok = True
            income_ok = len(income_overlap) == 0
        elif scenario == "s1_mix":
            city_ok = len(city_overlap) == 0
            income_ok = len(income_overlap) == 0
        else:
            city_ok = True
            income_ok = True

        print(f"\n[{scenario}] n={len(g)}")
        print(f"  cities unique={len(g_cities)}")
        print(f"  cities sample={sorted(g_cities)[:20]}")
        print(f"  income unique={len(g_income)}")
        print(f"  income min={int(g['income'].min())} max={int(g['income'].max())}")
        print(
            f"  n_rows min={g['n_rows'].min()} "
            f"median={int(g['n_rows'].median())} "
            f"max={g['n_rows'].max()}"
        )
        print(f"  city overlap={len(city_overlap)} {'OK' if city_ok else 'ERROR'}")
        print(f"  income overlap={len(income_overlap)} {'OK' if income_ok else 'ERROR'}")

        if not city_ok:
            print(f"    overlapping cities={sorted(city_overlap)[:20]}")
        if not income_ok:
            print(f"    overlapping incomes={sorted(income_overlap)[:20]}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q84 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q84.")
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
    for name in ['q84_normal', 'q84_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q84_normal"
    OUT_ANOM = args.output_dir / "q84_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(
        f"[CONFIG] Q84  "
        f"K={K}  "
        f"N_NORMAL={N_NORMAL}  "
        f"N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}"
    )
    print("[CONFIG] anomaly dimensions:")
    print("  - city")
    print("  - income")
    print(f"[CONFIG] generate_mix={GENERATE_MIX}")
    print(f"[CONFIG] output base={EXTERNAL_BASE_DIR}")

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

    print("\n[GEN] s1_city")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_city",
            N_ANOMALY_PER_TYPE,
            sampler.s1_city_params,
            con,
        )
    )

    print("\n[GEN] s1_income")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_income",
            N_ANOMALY_PER_TYPE,
            sampler.s1_income_params,
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

    print_diagnostics(normal_rows, anom_rows)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
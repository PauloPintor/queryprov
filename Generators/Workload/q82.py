#!/usr/bin/env python3
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

DB_PATH = None  # Set by command-line arguments.
DB_SCHEMA = "tpcds"

SEED = 42

N_NORMAL = 65_000
N_ANOMALY_PER_TYPE = 350

GENERATE_MIX = True

# K=0.5 -> sampling mais concentrado/normal mais compacto
# K=1.0 -> perfil normal mais largo / sampling empírico
# K>1   -> sampling mais uniforme
K = 1.0

YEAR = 2002

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.


# ============================================================
# PRICE CONFIG
# ============================================================

PRICE_LO = 0
PRICE_HI = 90
PRICE_WINDOW = 30


# ============================================================
# DATE CONFIG
# ============================================================
# Alinhado com Q5/Q20/Q98:
#   normal  = contexto recente
#   anomaly = contexto histórico disjunto
#
# A data amostrada é o início da janela.
# Como a query usa uma janela de 60 dias, o início máximo é:
#   context_max - 60 dias

DATE_WINDOW_DAYS = 60

NORMAL_CONTEXT_MIN = date(2002, 1, 1)
NORMAL_CONTEXT_MAX = date(2002, 6, 30)

ANOM_CONTEXT_MIN = date(2001, 7, 1)
ANOM_CONTEXT_MAX = date(2001, 12, 31)


# ============================================================
# MANUFACTURER CONFIG
# ============================================================

N_MANUFACT = 4


# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
select
    i_item_id,
    i_item_desc,
    i_current_price,
    array_agg(
        list_value(
            item.prov,
            inventory.prov,
            date_dim.prov,
            store_sales.prov
        )
    ) as prov
from item,
     inventory,
     date_dim,
     store_sales
where i_current_price between {price} and {price}+{price_window}
  and inv_item_sk = i_item_sk
  and d_date_sk = inv_date_sk
  and d_date between cast('{invdate}' as date)
                 and cast('{invdate}' as date) + interval {date_window_days} day
  and i_manufact_id in ({m1}, {m2}, {m3}, {m4})
  and inv_quantity_on_hand between 100 and 500
  and ss_item_sk = i_item_sk
group by
    i_item_id,
    i_item_desc,
    i_current_price
order by
    i_item_id
limit 100
"""


COUNT_TEMPLATE = """
select count(*) as n_rows
from (
    select
        i_item_id,
        i_item_desc,
        i_current_price
    from item,
         inventory,
         date_dim,
         store_sales
    where i_current_price between {price} and {price}+{price_window}
      and inv_item_sk = i_item_sk
      and d_date_sk = inv_date_sk
      and d_date between cast('{invdate}' as date)
                     and cast('{invdate}' as date) + interval {date_window_days} day
      and i_manufact_id in ({m1}, {m2}, {m3}, {m4})
      and inv_quantity_on_hand between 100 and 500
      and ss_item_sk = i_item_sk
    group by
        i_item_id,
        i_item_desc,
        i_current_price
) t
"""


# ============================================================
# HELPERS
# ============================================================

def _weights_from_counts(counts: np.ndarray, k: float) -> np.ndarray:
    alpha = 1.0 / max(k, 0.05)
    w = counts.astype(float) ** alpha
    w = w / w.sum()
    return w


def sample_weighted_without_replacement(
    pool: list[int],
    weights: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> list[int]:
    if len(pool) < n:
        raise ValueError(f"Pool demasiado pequeno: {len(pool)} < {n}")

    idx = rng.choice(
        len(pool),
        size=n,
        replace=False,
        p=weights,
    )

    return [pool[int(i)] for i in idx]


def date_to_idx(d: date) -> int:
    return d.toordinal()


def idx_to_date(i: int) -> date:
    return date.fromordinal(int(i))


def valid_sdate_range(
    context_min: date,
    context_max: date,
    window_days: int,
) -> tuple[date, date]:
    max_start = context_max - timedelta(days=window_days)

    if max_start < context_min:
        raise ValueError(
            f"Contexto temporal inválido: {context_min}..{context_max} "
            f"não suporta janela de {window_days} dias."
        )

    return context_min, max_start


def sample_recent_reflected_exponential(
    lo: date,
    hi: date,
    rng: np.random.Generator,
    scale_days: float | None = None,
) -> date:
    lo_i = date_to_idx(lo)
    hi_i = date_to_idx(hi)

    span = hi_i - lo_i + 1

    if scale_days is None:
        scale_days = max(span / 5.0, 1.0)

    for _ in range(100_000):
        offset = int(rng.exponential(scale_days))
        v = hi_i - offset

        if lo_i <= v <= hi_i:
            return idx_to_date(v)

    return hi


def sample_old_uniform(
    lo: date,
    hi: date,
    rng: np.random.Generator,
) -> date:
    lo_i = date_to_idx(lo)
    hi_i = date_to_idx(hi)

    return idx_to_date(
        int(rng.integers(lo_i, hi_i + 1))
    )


# ============================================================
# CONTINUOUS RANGE
# ============================================================

def compute_continuous_range(
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
            f"Sem região anómala para domínio [{lo}, {hi}] com K={k}."
        )

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


def sample_normal_price(
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


def sample_anomaly_price(
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
        raise ValueError("Sem região anómala.")

    return sample_truncated_normal(
        r["mu"],
        r["sigma"],
        lo,
        hi,
        rng,
    )


def print_price_range(r: dict) -> None:
    print("\n[PRICE]")
    print(f"  domain=[{r['lo']}, {r['hi']}]")
    print(f"  window={PRICE_WINDOW}")
    print(f"  mu={r['mu']:.2f} sigma={r['sigma']:.2f}")
    print(f"  normal=[{r['lo_normal']}, {r['hi_normal']}]")

    parts = []
    if r["has_left"]:
        parts.append(f"[{r['lo_anom_left']}, {r['hi_anom_left']}]")
    if r["has_right"]:
        parts.append(f"[{r['lo_anom_right']}, {r['hi_anom_right']}]")

    print(f"  anomaly={parts}")


# ============================================================
# MANUFACTURER TIERS
# ============================================================

def fit_manufacturer_tiers(
    con: duckdb.DuckDBPyConnection,
    k: float,
    rng: np.random.Generator,
) -> dict:
    """
    Q82 uses a set of four manufacturer identifiers.

    Although manufacturer identifiers are numerical ids, in this query they
    are used as a discrete set-valued categorical parameter. Therefore, we
    derive normal/anomalous pools empirically from the executable workload
    space rather than using a Gaussian model.
    """

    normal_lo, normal_hi = valid_sdate_range(
        NORMAL_CONTEXT_MIN,
        NORMAL_CONTEXT_MAX,
        DATE_WINDOW_DAYS,
    )

    df = con.execute(f"""
        select
            i.i_manufact_id as manufact_id,
            count(*) as n_rows
        from item i
        join inventory inv
          on inv.inv_item_sk = i.i_item_sk
        join date_dim d
          on d.d_date_sk = inv.inv_date_sk
        join store_sales ss
          on ss.ss_item_sk = i.i_item_sk
        where i.i_manufact_id is not null
          and inv.inv_quantity_on_hand between 100 and 500
          and d.d_date between cast('{normal_lo.isoformat()}' as date)
                           and cast('{normal_hi.isoformat()}' as date) + interval {DATE_WINDOW_DAYS} day
        group by i.i_manufact_id
        order by n_rows desc,
                 manufact_id
    """).df()

    if len(df) < 2 * N_MANUFACT:
        raise ValueError(
            f"Poucos manufacturers distintos: {len(df)}. "
            f"A Q82 precisa de pelo menos {N_MANUFACT} normais "
            f"e {N_MANUFACT} anómalos."
        )

    manufacturers = df["manufact_id"].astype(int).tolist()
    counts = df["n_rows"].astype(float).to_numpy()
    n = len(manufacturers)

    normal_size = n // 2
    normal_size = max(N_MANUFACT, min(normal_size, n - N_MANUFACT))

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

    normal_manufacturers = [manufacturers[i] for i in normal_idx]
    anomaly_manufacturers = [manufacturers[i] for i in anomaly_idx]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    normal_w = _weights_from_counts(normal_counts, k)
    anomaly_w = _weights_from_counts(anomaly_counts, k)

    alpha = 1.0 / max(k, 0.05)

    print("\n[MANUFACTURER]")
    print(f"  context=[{normal_lo}, {normal_hi}] + {DATE_WINDOW_DAYS} days")
    print(f"  n_manufacturers={n}  K={k}  alpha={alpha:.3f}")
    print(f"  normal n={len(normal_manufacturers)}")
    print(f"  anomaly n={len(anomaly_manufacturers)}")
    print(f"  normal sample={normal_manufacturers[:20]}")
    print(f"  anomaly sample={anomaly_manufacturers[:20]}")

    return {
        "normal": normal_manufacturers,
        "normal_w": normal_w,
        "anomaly": anomaly_manufacturers,
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

        self.price_range = compute_continuous_range(
            PRICE_LO,
            PRICE_HI,
            K,
        )

        print_price_range(self.price_range)

        self.normal_date_lo, self.normal_date_hi = valid_sdate_range(
            NORMAL_CONTEXT_MIN,
            NORMAL_CONTEXT_MAX,
            DATE_WINDOW_DAYS,
        )

        self.anom_date_lo, self.anom_date_hi = valid_sdate_range(
            ANOM_CONTEXT_MIN,
            ANOM_CONTEXT_MAX,
            DATE_WINDOW_DAYS,
        )

        print("\n[DATE]")
        print(f"  window_days={DATE_WINDOW_DAYS}")
        print(f"  normal context=[{NORMAL_CONTEXT_MIN}, {NORMAL_CONTEXT_MAX}]")
        print(f"  normal invdate=[{self.normal_date_lo}, {self.normal_date_hi}]")
        print(f"  anomaly context=[{ANOM_CONTEXT_MIN}, {ANOM_CONTEXT_MAX}]")
        print(f"  anomaly invdate=[{self.anom_date_lo}, {self.anom_date_hi}]")

        self.manufacturer = fit_manufacturer_tiers(con, K, rng)

    def normal_params(self) -> dict:
        return self._build(
            scenario="normal",
            label=0,
            price_mode="normal",
            date_mode="normal",
            manufact_mode="normal",
        )

    def s1_price_params(self) -> dict:
        return self._build(
            scenario="s1_price",
            label=1,
            price_mode="anomaly",
            date_mode="normal",
            manufact_mode="normal",
        )

    def s1_date_params(self) -> dict:
        return self._build(
            scenario="s1_date",
            label=1,
            price_mode="normal",
            date_mode="anomaly",
            manufact_mode="normal",
        )

    def s1_manufact_params(self) -> dict:
        return self._build(
            scenario="s1_manufact",
            label=1,
            price_mode="normal",
            date_mode="normal",
            manufact_mode="anomaly",
        )

    def s1_mix_params(self) -> dict:
        return self._build(
            scenario="s1_mix",
            label=1,
            price_mode="anomaly",
            date_mode="anomaly",
            manufact_mode="anomaly",
        )

    def _sample_price(self, mode: str) -> int:
        if mode == "normal":
            return sample_normal_price(self.price_range, self.rng)

        if mode == "anomaly":
            return sample_anomaly_price(self.price_range, self.rng)

        raise ValueError(f"Modo de price inválido: {mode}")

    def _sample_date(self, mode: str) -> date:
        if mode == "normal":
            return sample_recent_reflected_exponential(
                self.normal_date_lo,
                self.normal_date_hi,
                self.rng,
            )

        if mode == "anomaly":
            return sample_old_uniform(
                self.anom_date_lo,
                self.anom_date_hi,
                self.rng,
            )

        raise ValueError(f"Modo de date inválido: {mode}")

    def _sample_manufacturers(self, mode: str) -> list[int]:
        if mode == "normal":
            return sample_weighted_without_replacement(
                self.manufacturer["normal"],
                self.manufacturer["normal_w"],
                self.rng,
                N_MANUFACT,
            )

        if mode == "anomaly":
            return sample_weighted_without_replacement(
                self.manufacturer["anomaly"],
                self.manufacturer["anomaly_w"],
                self.rng,
                N_MANUFACT,
            )

        raise ValueError(f"Modo de manufacturer inválido: {mode}")

    def _build(
        self,
        scenario: str,
        label: int,
        price_mode: str,
        date_mode: str,
        manufact_mode: str,
    ) -> dict:
        price = self._sample_price(price_mode)
        invdate = self._sample_date(date_mode)
        manufacturers = self._sample_manufacturers(manufact_mode)

        return {
            "label": label,
            "scenario": scenario,

            "price": price,
            "price_window": PRICE_WINDOW,

            "invdate": invdate.isoformat(),
            "date_window_days": DATE_WINDOW_DAYS,

            "m1": manufacturers[0],
            "m2": manufacturers[1],
            "m3": manufacturers[2],
            "m4": manufacturers[3],

            "year": YEAR,
        }


# ============================================================
# RENDERING
# ============================================================

def render_sql(params: dict) -> str:
    return SQL_TEMPLATE.format(**params)


def render_count(params: dict) -> str:
    return COUNT_TEMPLATE.format(**params)


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

def manufact_set(df: pd.DataFrame) -> set[int]:
    return (
        set(df["m1"].astype(int))
        | set(df["m2"].astype(int))
        | set(df["m3"].astype(int))
        | set(df["m4"].astype(int))
    )


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

    if param == "price":
        norm_values = set(norm["price"].astype(int))
        anom_values = set(g["price"].astype(int))
    elif param == "date":
        norm_values = set(norm["invdate"].astype(str))
        anom_values = set(g["invdate"].astype(str))
    elif param == "manufact":
        norm_values = manufact_set(norm)
        anom_values = manufact_set(g)
    else:
        raise ValueError(param)

    overlap = norm_values & anom_values

    ok = len(overlap) == 0 if should_be_disjoint else True

    print(f"  {param:<10} overlap={len(overlap):<5} {'OK' if ok else 'ERROR'}")

    if not ok:
        sample = sorted(overlap)[:20]
        print(f"      overlap sample={sample}")


def print_diagnostics(
    normal_rows: list[dict],
    anom_rows: list[dict],
) -> None:
    norm = pd.DataFrame(normal_rows)
    anom = pd.DataFrame(anom_rows)

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    norm_manufact = manufact_set(norm)

    print(f"\n[NORMAL] n={len(norm)}")
    print(
        f"  price unique={norm['price'].nunique()} "
        f"min={int(norm['price'].min())} "
        f"max={int(norm['price'].max())}"
    )
    print(
        f"  invdate unique={norm['invdate'].nunique()} "
        f"min={norm['invdate'].min()} "
        f"max={norm['invdate'].max()}"
    )
    print(f"  manufacturers unique={len(norm_manufact)}")
    print(f"  year unique={sorted(norm['year'].unique().tolist())}")
    print(
        f"  n_rows min={norm['n_rows'].min()} "
        f"median={int(norm['n_rows'].median())} "
        f"max={norm['n_rows'].max()}"
    )

    print(f"\n[ANOMALY] n={len(anom)}")
    print(f"  scenarios={anom['scenario'].value_counts().to_dict()}")

    for scenario in sorted(anom["scenario"].unique()):
        g = anom[anom["scenario"] == scenario]
        g_manufact = manufact_set(g)

        print(f"\n[{scenario}] n={len(g)}")
        print(
            f"  price unique={g['price'].nunique()} "
            f"min={int(g['price'].min())} "
            f"max={int(g['price'].max())}"
        )
        print(
            f"  invdate unique={g['invdate'].nunique()} "
            f"min={g['invdate'].min()} "
            f"max={g['invdate'].max()}"
        )
        print(f"  manufacturers unique={len(g_manufact)}")
        print(f"  year unique={sorted(g['year'].unique().tolist())}")
        print(
            f"  n_rows min={g['n_rows'].min()} "
            f"median={int(g['n_rows'].median())} "
            f"max={g['n_rows'].max()}"
        )

        if scenario == "s1_price":
            print_overlap_for_param(norm, anom, scenario, "price", True)
            print_overlap_for_param(norm, anom, scenario, "date", False)
            print_overlap_for_param(norm, anom, scenario, "manufact", False)

        elif scenario == "s1_date":
            print_overlap_for_param(norm, anom, scenario, "price", False)
            print_overlap_for_param(norm, anom, scenario, "date", True)
            print_overlap_for_param(norm, anom, scenario, "manufact", False)

        elif scenario == "s1_manufact":
            print_overlap_for_param(norm, anom, scenario, "price", False)
            print_overlap_for_param(norm, anom, scenario, "date", False)
            print_overlap_for_param(norm, anom, scenario, "manufact", True)

        elif scenario == "s1_mix":
            print_overlap_for_param(norm, anom, scenario, "price", True)
            print_overlap_for_param(norm, anom, scenario, "date", True)
            print_overlap_for_param(norm, anom, scenario, "manufact", True)

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q82 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q82.")
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
    for name in ['q82_normal', 'q82_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q82_normal"
    OUT_ANOM = args.output_dir / "q82_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(
        f"[CONFIG] Q82  "
        f"K={K}  "
        f"N_NORMAL={N_NORMAL}  "
        f"N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}"
    )
    print("[CONFIG] anomaly dimensions:")
    print("  - price")
    print("  - inventory date")
    print("  - manufacturer set")
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

    print("\n[GEN] s1_price")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_price",
            N_ANOMALY_PER_TYPE,
            sampler.s1_price_params,
            con,
        )
    )

    print("\n[GEN] s1_date")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_date",
            N_ANOMALY_PER_TYPE,
            sampler.s1_date_params,
            con,
        )
    )

    print("\n[GEN] s1_manufact")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_manufact",
            N_ANOMALY_PER_TYPE,
            sampler.s1_manufact_params,
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
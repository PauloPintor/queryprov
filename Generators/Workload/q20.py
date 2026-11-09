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

# K=0.5 -> distribuição mais concentrada
# K=1.0 -> distribuição empírica
# K>1   -> mais uniforme
K = 1.0

# Q20 usa uma janela de 30 dias:
# d_date between SDATE and SDATE + INTERVAL 30 DAY
DATE_WINDOW_DAYS = 30

# Contextos temporais globais.
# O gerador escolhe SDATE de forma a que a janela completa
# fique dentro destes contextos.
NORMAL_CONTEXT_MIN = date(2002, 1, 1)
NORMAL_CONTEXT_MAX = date(2002, 6, 30)

ANOM_CONTEXT_MIN = date(2001, 7, 1)
ANOM_CONTEXT_MAX = date(2001, 12, 31)

RECENCY_EXP_SCALE = 0.25

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.


# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
select i_item_id
      ,i_item_desc
      ,i_category
      ,i_class
      ,i_current_price
      ,sum(cs_ext_sales_price) as itemrevenue
      ,sum(cs_ext_sales_price)*100/sum(sum(cs_ext_sales_price)) over
          (partition by i_class) as revenueratio
      ,array_agg(list_value(catalog_sales.prov, item.prov, date_dim.prov)) as prov
from catalog_sales
    ,item
    ,date_dim
where cs_item_sk = i_item_sk
  and i_category in ('{category1}', '{category2}', '{category3}')
  and cs_sold_date_sk = d_date_sk
  and d_date between cast('{sdate}' as date)
                 and cast('{sdate}' as date) + INTERVAL {window_days} DAY
group by i_item_id
        ,i_item_desc
        ,i_category
        ,i_class
        ,i_current_price
order by i_category
        ,i_class
        ,i_item_id
        ,i_item_desc
        ,revenueratio
limit 100
"""


COUNT_TEMPLATE = """
select count(*) from (
    select i_item_id
    from catalog_sales
        ,item
        ,date_dim
    where cs_item_sk = i_item_sk
      and i_category in ('{category1}', '{category2}', '{category3}')
      and cs_sold_date_sk = d_date_sk
      and d_date between cast('{sdate}' as date)
                     and cast('{sdate}' as date) + INTERVAL {window_days} DAY
    group by i_item_id
            ,i_item_desc
            ,i_category
            ,i_class
            ,i_current_price
) t
"""


# ============================================================
# CATEGORY
# ============================================================

def fit_category_tiers(
    con: duckdb.DuckDBPyConnection,
    k: float,
    rng: np.random.Generator,
) -> dict:
    """
    Categorias são tratadas como domínio categórico.

    Estratégia:
      - calcula frequência empírica no espaço executável da Q20;
      - cria pools normal/anómalo disjuntos;
      - para 10 categorias, fica tipicamente 5 normal / 5 anómalo;
      - K controla a concentração da amostragem dentro dos pools.
    """

    df = con.execute("""
        SELECT i.i_category AS category, COUNT(*) AS n_sales
        FROM catalog_sales cs
        JOIN item i ON cs.cs_item_sk = i.i_item_sk
        WHERE i.i_category IS NOT NULL
        GROUP BY i.i_category
        ORDER BY n_sales DESC, category
    """).df()

    if len(df) < 6:
        raise ValueError(f"Poucas categorias distintas: {len(df)}")

    cats = df["category"].astype(str).tolist()
    counts = df["n_sales"].astype(float).to_numpy()
    n = len(cats)

    # Domínio pequeno: pools equilibrados e disjuntos.
    # Como cada query precisa de 3 categorias, cada pool precisa de >= 3.
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

    normal_cats = [cats[i] for i in normal_idx]
    anomaly_cats = [cats[i] for i in anomaly_idx]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    alpha = 1.0 / max(k, 0.05)

    normal_w = normal_counts**alpha
    normal_w = normal_w / normal_w.sum()

    anomaly_w = anomaly_counts**alpha
    anomaly_w = anomaly_w / anomaly_w.sum()

    print("\n[CATEGORY]")
    print(f"  n_categories={n}  K={k}  alpha={alpha:.3f}")
    print(f"  normal n={len(normal_cats)}  anomaly n={len(anomaly_cats)}")
    print(f"  normal={normal_cats}")
    print(f"  anomaly={anomaly_cats}")

    return {
        "normal": normal_cats,
        "normal_w": normal_w,
        "anomaly": anomaly_cats,
        "anomaly_w": anomaly_w,
        "df": df,
    }


def sample_categories(
    pool: list[str],
    weights: np.ndarray,
    rng: np.random.Generator,
    n: int = 3,
) -> list[str]:
    if len(pool) < n:
        raise ValueError(f"Pool de categorias tem {len(pool)}, precisa de {n}.")

    idx = rng.choice(
        len(pool),
        size=n,
        replace=False,
        p=weights,
    )

    return [pool[int(i)] for i in idx]


# ============================================================
# DATE
# ============================================================

def valid_sdate_range(
    context_min: date,
    context_max: date,
    window_days: int,
) -> tuple[date, date]:
    """
    Define o intervalo válido para SDATE.

    Se a query acede a:
        [SDATE, SDATE + window_days]

    então SDATE só pode ser amostrado até:
        context_max - window_days

    Isto garante que a janela completa da query fica dentro do
    contexto temporal normal ou anómalo.
    """

    sdate_min = context_min
    sdate_max = context_max - timedelta(days=window_days)

    if sdate_max < sdate_min:
        raise ValueError(
            f"Contexto temporal [{context_min}, {context_max}] "
            f"é demasiado pequeno para uma janela de {window_days} dias."
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
    """
    Amostra um inteiro em [lo, hi], concentrado perto de hi.

    Isto modela o viés de recência no comportamento normal.
    """

    width = hi - lo

    if width <= 0:
        return hi

    for _ in range(100_000):
        u = rng.exponential(scale=scale)
        if 0 <= u <= 1:
            return int(round(hi - u * width))

    return hi


def sample_old_uniform(
    lo: int,
    hi: int,
    rng: np.random.Generator,
) -> int:
    """
    Amostra uniformemente no contexto temporal anómalo.
    """

    return int(rng.integers(lo, hi + 1))


# ============================================================
# SAMPLER
# ============================================================

class QuerySampler:
    def __init__(self, con: duckdb.DuckDBPyConnection, rng: np.random.Generator):
        self.rng = rng

        self.category = fit_category_tiers(con, K, rng)

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
        print("  normal sampling = reflected exponential towards most recent valid SDATE")
        print("  anomaly sampling = uniform over older valid SDATE")

    def normal_params(self) -> dict:
        cats = sample_categories(
            self.category["normal"],
            self.category["normal_w"],
            self.rng,
            3,
        )

        d_idx = sample_recent_reflected_exponential(
            self.normal_lo,
            self.normal_hi,
            self.rng,
        )

        d = idx_to_date(d_idx, self.normal_sdate_min)

        return self._params(
            scenario="normal",
            label=0,
            cats=cats,
            sdate=d,
            date_idx=d_idx,
        )

    def s1_category_params(self) -> dict:
        cats = sample_categories(
            self.category["anomaly"],
            self.category["anomaly_w"],
            self.rng,
            3,
        )

        d_idx = sample_recent_reflected_exponential(
            self.normal_lo,
            self.normal_hi,
            self.rng,
        )

        d = idx_to_date(d_idx, self.normal_sdate_min)

        return self._params(
            scenario="s1_category",
            label=1,
            cats=cats,
            sdate=d,
            date_idx=d_idx,
        )

    def s1_date_params(self) -> dict:
        cats = sample_categories(
            self.category["normal"],
            self.category["normal_w"],
            self.rng,
            3,
        )

        d_idx = sample_old_uniform(
            self.anom_lo,
            self.anom_hi,
            self.rng,
        )

        d = idx_to_date(d_idx, self.anom_sdate_min)

        return self._params(
            scenario="s1_date",
            label=1,
            cats=cats,
            sdate=d,
            date_idx=d_idx,
        )

    def s1_mix_params(self) -> dict:
        cats = sample_categories(
            self.category["anomaly"],
            self.category["anomaly_w"],
            self.rng,
            3,
        )

        d_idx = sample_old_uniform(
            self.anom_lo,
            self.anom_hi,
            self.rng,
        )

        d = idx_to_date(d_idx, self.anom_sdate_min)

        return self._params(
            scenario="s1_mix",
            label=1,
            cats=cats,
            sdate=d,
            date_idx=d_idx,
        )

    def _params(
        self,
        scenario: str,
        label: int,
        cats: list[str],
        sdate: date,
        date_idx: int,
    ) -> dict:
        return {
            "label": label,
            "scenario": scenario,
            "category1": cats[0],
            "category2": cats[1],
            "category3": cats[2],
            "sdate": sdate.isoformat(),
            "date_idx": date_idx,
            "window_days": DATE_WINDOW_DAYS,
            "window_start": sdate.isoformat(),
            "window_end": (sdate + timedelta(days=DATE_WINDOW_DAYS)).isoformat(),
        }


# ============================================================
# WRITER
# ============================================================

def _sql_quote_escape(s: str) -> str:
    return str(s).replace("'", "''")


def _render_sql(params: dict) -> str:
    p = dict(params)
    p["category1"] = _sql_quote_escape(p["category1"])
    p["category2"] = _sql_quote_escape(p["category2"])
    p["category3"] = _sql_quote_escape(p["category3"])
    return SQL_TEMPLATE.format(**p)


def _render_count(params: dict) -> str:
    p = dict(params)
    p["category1"] = _sql_quote_escape(p["category1"])
    p["category2"] = _sql_quote_escape(p["category2"])
    p["category3"] = _sql_quote_escape(p["category3"])
    return COUNT_TEMPLATE.format(**p)


def _write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


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

def _cat_set(df: pd.DataFrame) -> set[str]:
    return set(df["category1"]) | set(df["category2"]) | set(df["category3"])


def print_diagnostics(normal_rows: list[dict], anom_rows: list[dict]) -> None:
    norm = pd.DataFrame(normal_rows)
    anom = pd.DataFrame(anom_rows)

    norm_cats = _cat_set(norm)
    norm_dates = set(norm["sdate"])
    norm_windows = set(zip(norm["window_start"], norm["window_end"]))

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    print(f"\n[NORMAL] n={len(norm)}")
    print(f"  categories unique={len(norm_cats)}")
    print(f"  dates unique={len(norm_dates)}")
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

    for scenario in sorted(anom["scenario"].unique()):
        g = anom[anom["scenario"] == scenario]

        g_cats = _cat_set(g)
        g_dates = set(g["sdate"])
        g_windows = set(zip(g["window_start"], g["window_end"]))

        cat_overlap = norm_cats & g_cats
        date_overlap = norm_dates & g_dates
        window_overlap = norm_windows & g_windows

        if scenario == "s1_category":
            cat_ok = len(cat_overlap) == 0
            date_ok = True
            window_ok = True
        elif scenario == "s1_date":
            cat_ok = True
            date_ok = len(date_overlap) == 0
            window_ok = len(window_overlap) == 0
        else:
            cat_ok = len(cat_overlap) == 0
            date_ok = len(date_overlap) == 0
            window_ok = len(window_overlap) == 0

        print(f"\n[{scenario}] n={len(g)}")
        print(f"  categories unique={len(g_cats)}")
        print(f"  dates unique={len(g_dates)}")
        print(f"  sdate min={g['sdate'].min()} max={g['sdate'].max()}")
        print(f"  window_start min={g['window_start'].min()} max={g['window_start'].max()}")
        print(f"  window_end   min={g['window_end'].min()} max={g['window_end'].max()}")
        print(
            f"  n_rows min={g['n_rows'].min()} "
            f"median={int(g['n_rows'].median())} "
            f"max={g['n_rows'].max()}"
        )

        print(f"  category overlap={len(cat_overlap)} {'OK' if cat_ok else 'ERROR'}")
        print(f"  date overlap={len(date_overlap)} {'OK' if date_ok else 'ERROR'}")
        print(f"  window overlap={len(window_overlap)} {'OK' if window_ok else 'ERROR'}")

        if not cat_ok:
            print(f"    overlapping categories={sorted(cat_overlap)}")

        if not date_ok:
            print(f"    overlapping dates={sorted(date_overlap)[:20]}")

        if not window_ok:
            print(f"    overlapping windows={sorted(window_overlap)[:20]}")

    print("\n[EXPECTED TEMPORAL CONTEXTS]")
    print(f"  normal context : [{NORMAL_CONTEXT_MIN}, {NORMAL_CONTEXT_MAX}]")
    print(f"  anomaly context: [{ANOM_CONTEXT_MIN}, {ANOM_CONTEXT_MAX}]")

    norm_window_end_max = pd.to_datetime(norm["window_end"]).max().date()
    anom_window_end_max = pd.to_datetime(anom["window_end"]).max().date()

    print("\n[WINDOW CHECK]")
    print(
        f"  normal max window_end={norm_window_end_max} "
        f"{'OK' if norm_window_end_max <= NORMAL_CONTEXT_MAX else 'ERROR'}"
    )

    if len(anom) > 0:
        date_anom = anom[anom["scenario"].isin(["s1_date", "s1_mix"])]
        if len(date_anom) > 0:
            date_anom_window_end_max = pd.to_datetime(date_anom["window_end"]).max().date()
            print(
                f"  date-anomaly max window_end={date_anom_window_end_max} "
                f"{'OK' if date_anom_window_end_max <= ANOM_CONTEXT_MAX else 'ERROR'}"
            )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q20 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q20.")
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
    for name in ['q20_normal', 'q20_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q20_normal"
    OUT_ANOM = args.output_dir / "q20_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(
        f"[CONFIG] Q20  "
        f"K={K}  "
        f"N_NORMAL={N_NORMAL}  "
        f"N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}"
    )
    print("[CONFIG] fact table = catalog_sales")
    print("[CONFIG] normal date = reflected exponential over recent context")
    print("[CONFIG] anomaly date = uniform over older context")
    print("[CONFIG] full query window is constrained to remain inside its context")
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

    anom_all: list[dict] = []

    print("\n[GEN] s1_category")
    anom_all.extend(
        write_queries(
            OUT_ANOM,
            "s1_category",
            N_ANOMALY_PER_TYPE,
            sampler.s1_category_params,
            con,
        )
    )

    print("\n[GEN] s1_date")
    anom_all.extend(
        write_queries(
            OUT_ANOM,
            "s1_date",
            N_ANOMALY_PER_TYPE,
            sampler.s1_date_params,
            con,
        )
    )

    if GENERATE_MIX:
        print("\n[GEN] s1_mix")
        anom_all.extend(
            write_queries(
                OUT_ANOM,
                "s1_mix",
                N_ANOMALY_PER_TYPE,
                sampler.s1_mix_params,
                con,
            )
        )

    _write_metadata(OUT_ANOM / "metadata.csv", anom_all)

    con.close()

    print_diagnostics(normal_rows, anom_all)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
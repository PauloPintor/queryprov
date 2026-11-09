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

K = 1.0

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.


# ============================================================
# DATE CONFIG
# ============================================================
# Q98 usa uma janela temporal de 30 dias.
#
# Normal:
#   contexto recente em 2002
#
# Anomaly:
#   contexto histórico disjunto em 2001

DATE_WINDOW_DAYS = 30

NORMAL_CONTEXT_MIN = date(2002, 1, 1)
NORMAL_CONTEXT_MAX = date(2002, 6, 30)

ANOM_CONTEXT_MIN = date(2001, 7, 1)
ANOM_CONTEXT_MAX = date(2001, 12, 31)


# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
select i_item_id
      ,i_item_desc
      ,i_category
      ,i_class
      ,i_current_price
      ,sum(ss_ext_sales_price) as itemrevenue
      ,sum(ss_ext_sales_price)*100/sum(sum(ss_ext_sales_price)) over
          (partition by i_class) as revenueratio
      ,array_agg(list_value(store_sales.prov, item.prov, date_dim.prov)) as prov
from store_sales
    ,item
    ,date_dim
where ss_item_sk = i_item_sk
  and i_category in ('{category1}', '{category2}', '{category3}')
  and ss_sold_date_sk = d_date_sk
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
"""


COUNT_TEMPLATE = """
select count(*) as n_rows
from (
    select i_item_id
    from store_sales
        ,item
        ,date_dim
    where ss_item_sk = i_item_sk
      and i_category in ('{category1}', '{category2}', '{category3}')
      and ss_sold_date_sk = d_date_sk
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
    Frequências das categorias no espaço de store_sales + item.

    A query recebe três categorias. Por isso garantimos pelo menos três
    categorias no pool normal e três no pool anómalo.
    """

    df = con.execute("""
        SELECT
            i.i_category AS category,
            COUNT(*) AS n_sales
        FROM store_sales ss
        JOIN item i
          ON ss.ss_item_sk = i.i_item_sk
        WHERE i.i_category IS NOT NULL
        GROUP BY i.i_category
        ORDER BY n_sales DESC, category
    """).df()

    if len(df) < 6:
        raise ValueError(f"Poucas categorias distintas: {len(df)}")

    cats = df["category"].astype(str).tolist()
    counts = df["n_sales"].astype(float).to_numpy()
    n = len(cats)

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

    anomaly_idx = [
        i
        for i in range(n)
        if i not in normal_set
    ]

    normal_cats = [
        cats[i]
        for i in normal_idx
    ]

    anomaly_cats = [
        cats[i]
        for i in anomaly_idx
    ]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    alpha = 1.0 / max(k, 0.05)

    normal_w = normal_counts ** alpha
    normal_w = normal_w / normal_w.sum()

    anomaly_w = anomaly_counts ** alpha
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
        raise ValueError(
            f"Pool de categorias tem {len(pool)}, precisa de {n}."
        )

    idx = rng.choice(
        len(pool),
        size=n,
        replace=False,
        p=weights,
    )

    return [
        pool[int(i)]
        for i in idx
    ]


# ============================================================
# DATE HELPERS
# ============================================================

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


def date_to_idx(d: date) -> int:
    return d.toordinal()


def idx_to_date(i: int) -> date:
    return date.fromordinal(int(i))


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
        print(f"  DATE_WINDOW_DAYS={DATE_WINDOW_DAYS}")
        print(f"  normal context=[{NORMAL_CONTEXT_MIN}, {NORMAL_CONTEXT_MAX}]")
        print(f"  normal sdate=[{self.normal_date_lo}, {self.normal_date_hi}]")
        print(f"  anomaly context=[{ANOM_CONTEXT_MIN}, {ANOM_CONTEXT_MAX}]")
        print(f"  anomaly sdate=[{self.anom_date_lo}, {self.anom_date_hi}]")

    def normal_params(self) -> dict:
        cats = sample_categories(
            self.category["normal"],
            self.category["normal_w"],
            self.rng,
            3,
        )

        sdate = sample_recent_reflected_exponential(
            self.normal_date_lo,
            self.normal_date_hi,
            self.rng,
        )

        return self._params(
            "normal",
            0,
            cats,
            sdate,
        )

    def s1_category_params(self) -> dict:
        cats = sample_categories(
            self.category["anomaly"],
            self.category["anomaly_w"],
            self.rng,
            3,
        )

        sdate = sample_recent_reflected_exponential(
            self.normal_date_lo,
            self.normal_date_hi,
            self.rng,
        )

        return self._params(
            "s1_category",
            1,
            cats,
            sdate,
        )

    def s1_date_params(self) -> dict:
        cats = sample_categories(
            self.category["normal"],
            self.category["normal_w"],
            self.rng,
            3,
        )

        sdate = sample_old_uniform(
            self.anom_date_lo,
            self.anom_date_hi,
            self.rng,
        )

        return self._params(
            "s1_date",
            1,
            cats,
            sdate,
        )

    def s1_mix_params(self) -> dict:
        cats = sample_categories(
            self.category["anomaly"],
            self.category["anomaly_w"],
            self.rng,
            3,
        )

        sdate = sample_old_uniform(
            self.anom_date_lo,
            self.anom_date_hi,
            self.rng,
        )

        return self._params(
            "s1_mix",
            1,
            cats,
            sdate,
        )

    def _params(
        self,
        scenario: str,
        label: int,
        cats: list[str],
        sdate: date,
    ) -> dict:
        return {
            "label": label,
            "scenario": scenario,
            "category1": cats[0],
            "category2": cats[1],
            "category3": cats[2],
            "sdate": sdate.isoformat(),
            "window_days": DATE_WINDOW_DAYS,
        }


# ============================================================
# RENDERING
# ============================================================

def _sql_quote_escape(s: str) -> str:
    return str(s).replace("'", "''")


def _prepare_sql_params(params: dict) -> dict:
    p = dict(params)
    p["category1"] = _sql_quote_escape(p["category1"])
    p["category2"] = _sql_quote_escape(p["category2"])
    p["category3"] = _sql_quote_escape(p["category3"])
    return p


def _render_sql(params: dict) -> str:
    return SQL_TEMPLATE.format(**_prepare_sql_params(params))


def _render_count(params: dict) -> str:
    return COUNT_TEMPLATE.format(**_prepare_sql_params(params))


def _write_metadata(
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
    sampler_fn,
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
                _render_count(params)
            ).fetchone()[0]
        except Exception as exc:
            print(f"[ERROR] count failed for {params}: {exc}")
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
                "n_rows": int(n_rows),
            }
        )

        i += 1

        if i % 1000 == 0:
            print(f"  [GEN] {prefix}: {i}/{n}")

    _write_metadata(
        out_dir / "metadata.csv",
        rows,
    )

    print(f"  [OK] {prefix}: {len(rows)} queries, attempts={attempts}")

    return rows


# ============================================================
# DIAGNOSTICS
# ============================================================

def _cat_set(df: pd.DataFrame) -> set[str]:
    return (
        set(df["category1"].astype(str))
        | set(df["category2"].astype(str))
        | set(df["category3"].astype(str))
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

    if param == "category":
        norm_values = _cat_set(norm)
        anom_values = _cat_set(g)
    elif param == "date":
        norm_values = set(norm["sdate"].astype(str))
        anom_values = set(g["sdate"].astype(str))
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
    print(f"  categories unique={len(_cat_set(norm))}")
    print(f"  dates unique={norm['sdate'].nunique()}")
    print(f"  sdate min={norm['sdate'].min()} max={norm['sdate'].max()}")
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
        print(f"  categories unique={len(_cat_set(g))}")
        print(f"  dates unique={g['sdate'].nunique()}")
        print(f"  sdate min={g['sdate'].min()} max={g['sdate'].max()}")
        print(
            f"  n_rows min={g['n_rows'].min()} "
            f"median={int(g['n_rows'].median())} "
            f"max={g['n_rows'].max()}"
        )

        if scenario == "s1_category":
            print_overlap_for_param(norm, anom, scenario, "category", True)
            print_overlap_for_param(norm, anom, scenario, "date", False)

        elif scenario == "s1_date":
            print_overlap_for_param(norm, anom, scenario, "category", False)
            print_overlap_for_param(norm, anom, scenario, "date", True)

        elif scenario == "s1_mix":
            print_overlap_for_param(norm, anom, scenario, "category", True)
            print_overlap_for_param(norm, anom, scenario, "date", True)

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q98 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q98.")
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
    for name in ['q98_normal', 'q98_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q98_normal"
    OUT_ANOM = args.output_dir / "q98_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(
        f"[CONFIG] Q98  "
        f"K={K}  "
        f"N_NORMAL={N_NORMAL}  "
        f"N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}"
    )
    print("[CONFIG] fact table = store_sales")
    print("[CONFIG] categorical parameter = i_category")
    print("[CONFIG] date normal = reflected exponential over recent 2002 context")
    print("[CONFIG] date anomaly = uniform over historical 2001 context")
    print(f"[CONFIG] DATE_WINDOW_DAYS={DATE_WINDOW_DAYS}")
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

    _write_metadata(
        OUT_ANOM / "metadata.csv",
        anom_all,
    )

    con.close()

    print_diagnostics(
        normal_rows,
        anom_all,
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
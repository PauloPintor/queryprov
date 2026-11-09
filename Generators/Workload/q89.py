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

# K=0.5 -> sampling mais concentrado nos pares frequentes
# K=1.0 -> sampling empírico
# K>1   -> sampling mais uniforme
K = 1.0

NORMAL_YEAR = 2002

N_PAIRS = 6

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.


# ============================================================
# SQL
# ============================================================

PAIR_PREDICATE_TEMPLATE = """
          and (
                 (i_category = '{cat1}' and i_class = '{class1}')
              or (i_category = '{cat2}' and i_class = '{class2}')
              or (i_category = '{cat3}' and i_class = '{class3}')
              or (i_category = '{cat4}' and i_class = '{class4}')
              or (i_category = '{cat5}' and i_class = '{class5}')
              or (i_category = '{cat6}' and i_class = '{class6}')
          )
"""


SQL_TEMPLATE = """
select *
from (
    select
        i_category,
        i_class,
        i_brand,
        s_store_name,
        s_company_name,
        d_moy,
        sum(ss_sales_price) as sum_sales,
        avg(sum(ss_sales_price)) over
          (partition by i_category, i_brand, s_store_name, s_company_name)
          as avg_monthly_sales,
        array_agg(
            list_value(
                item.prov,
                store_sales.prov,
                date_dim.prov,
                store.prov
            )
        ) as prov
    from item,
         store_sales,
         date_dim,
         store
    where ss_item_sk = i_item_sk
      and ss_sold_date_sk = d_date_sk
      and ss_store_sk = s_store_sk
      and d_year = {year}
{pair_predicate}
    group by
        i_category,
        i_class,
        i_brand,
        s_store_name,
        s_company_name,
        d_moy
) tmp1
where case
        when avg_monthly_sales <> 0
        then abs(sum_sales - avg_monthly_sales) / avg_monthly_sales
        else null
      end > 0.1
order by
    sum_sales - avg_monthly_sales,
    s_store_name
limit 100
"""


COUNT_TEMPLATE = """
select count(*) as n_rows
from (
    select *
    from (
        select
            i_category,
            i_class,
            i_brand,
            s_store_name,
            s_company_name,
            d_moy,
            sum(ss_sales_price) as sum_sales,
            avg(sum(ss_sales_price)) over
              (partition by i_category, i_brand, s_store_name, s_company_name)
              as avg_monthly_sales
        from item,
             store_sales,
             date_dim,
             store
        where ss_item_sk = i_item_sk
          and ss_sold_date_sk = d_date_sk
          and ss_store_sk = s_store_sk
          and d_year = {year}
{pair_predicate}
        group by
            i_category,
            i_class,
            i_brand,
            s_store_name,
            s_company_name,
            d_moy
    ) tmp1
    where case
            when avg_monthly_sales <> 0
            then abs(sum_sales - avg_monthly_sales) / avg_monthly_sales
            else null
          end > 0.1
) t
"""


# ============================================================
# HELPERS
# ============================================================

def sql_escape(s: str) -> str:
    return str(s).replace("'", "''")


def _weights_from_counts(
    counts: np.ndarray,
    k: float,
) -> np.ndarray:
    alpha = 1.0 / max(k, 0.05)
    w = counts.astype(float) ** alpha
    w = w / w.sum()
    return w


def sample_weighted_without_replacement(
    pool: list[tuple[str, str]],
    weights: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> list[tuple[str, str]]:
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
# CATEGORY-CLASS TIERS
# ============================================================

def fit_category_class_tiers(
    con: duckdb.DuckDBPyConnection,
    k: float,
    rng: np.random.Generator,
) -> dict:
    """
    Q89 uses six item category-class pairs.

    The category-class frequencies are estimated in the same fixed year used
    by the generated workload, so the pools reflect the executable workload
    context rather than the full database.
    """

    df = con.execute(f"""
        select
            i.i_category as category,
            i.i_class as class,
            count(*) as n_sales
        from store_sales ss
        join item i
          on ss.ss_item_sk = i.i_item_sk
        join date_dim d
          on ss.ss_sold_date_sk = d.d_date_sk
        where i.i_category is not null
          and i.i_class is not null
          and d.d_year = {NORMAL_YEAR}
        group by
            i.i_category,
            i.i_class
        order by
            n_sales desc,
            category,
            class
    """).df()

    if len(df) < 2 * N_PAIRS:
        raise ValueError(
            f"Poucos pares category/class: {len(df)}. "
            f"Precisas de pelo menos {2 * N_PAIRS}."
        )

    pairs = list(
        zip(
            df["category"].astype(str),
            df["class"].astype(str),
        )
    )
    counts = df["n_sales"].astype(float).to_numpy()
    n = len(pairs)

    # Pool size is independent of K.
    # K controls only the concentration of sampling weights.
    normal_size = n // 2
    normal_size = max(N_PAIRS, min(normal_size, n - N_PAIRS))

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

    normal_pairs = [pairs[i] for i in normal_idx]
    anomaly_pairs = [pairs[i] for i in anomaly_idx]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    normal_w = _weights_from_counts(normal_counts, k)
    anomaly_w = _weights_from_counts(anomaly_counts, k)

    alpha = 1.0 / max(k, 0.05)

    print("\n[CATEGORY-CLASS]")
    print(f"  year={NORMAL_YEAR}")
    print(f"  total_pairs={n}")
    print(f"  K={k} alpha={alpha:.3f}")
    print(f"  normal n={len(normal_pairs)}")
    print(f"  anomaly n={len(anomaly_pairs)}")
    print(f"  normal sample={normal_pairs[:10]}")
    print(f"  anomaly sample={anomaly_pairs[:10]}")

    return {
        "normal": normal_pairs,
        "normal_w": normal_w,
        "anomaly": anomaly_pairs,
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
        self.pairs = fit_category_class_tiers(
            con,
            K,
            rng,
        )

    def normal_params(self) -> dict:
        pairs = sample_weighted_without_replacement(
            self.pairs["normal"],
            self.pairs["normal_w"],
            self.rng,
            N_PAIRS,
        )

        return self._build(
            scenario="normal",
            label=0,
            pairs=pairs,
        )

    def s1_category_class_params(self) -> dict:
        pairs = sample_weighted_without_replacement(
            self.pairs["anomaly"],
            self.pairs["anomaly_w"],
            self.rng,
            N_PAIRS,
        )

        return self._build(
            scenario="s1_category_class",
            label=1,
            pairs=pairs,
        )

    def _build(
        self,
        scenario: str,
        label: int,
        pairs: list[tuple[str, str]],
    ) -> dict:
        if len(set(pairs)) != len(pairs):
            raise ValueError("Repetição inesperada de pares category/class.")

        params = {
            "label": label,
            "scenario": scenario,
            "year": NORMAL_YEAR,
        }

        # Metadata keeps raw values.
        # Escaping is applied only at SQL rendering time.
        for i, (cat, cls) in enumerate(pairs, start=1):
            params[f"cat{i}"] = cat
            params[f"class{i}"] = cls

        return params


# ============================================================
# RENDERING
# ============================================================

def _prepare_sql_params(params: dict) -> dict:
    p = dict(params)

    for i in range(1, N_PAIRS + 1):
        p[f"cat{i}"] = sql_escape(p[f"cat{i}"])
        p[f"class{i}"] = sql_escape(p[f"class{i}"])

    p["pair_predicate"] = PAIR_PREDICATE_TEMPLATE.format(**p)

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

def pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()

    for i in range(1, N_PAIRS + 1):
        out |= set(
            zip(
                df[f"cat{i}"].astype(str),
                df[f"class{i}"].astype(str),
            )
        )

    return out


def print_diagnostics(
    normal_rows: list[dict],
    anom_rows: list[dict],
) -> None:
    norm = pd.DataFrame(normal_rows)
    anom = pd.DataFrame(anom_rows)

    norm_pairs = pair_set(norm)
    anom_pairs = pair_set(anom)

    overlap = norm_pairs & anom_pairs

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    print(f"\n[NORMAL] n={len(norm)}")
    print(f"  category-class pairs unique={len(norm_pairs)}")
    print(f"  pair sample={sorted(norm_pairs)[:20]}")
    print(f"  year unique={sorted(norm['year'].unique().tolist())}")
    print(
        f"  n_rows min={norm['n_rows'].min()} "
        f"median={int(norm['n_rows'].median())} "
        f"max={norm['n_rows'].max()}"
    )

    print(f"\n[ANOMALY] n={len(anom)}")
    print(f"  scenarios={anom['scenario'].value_counts().to_dict()}")
    print(f"  category-class pairs unique={len(anom_pairs)}")
    print(f"  pair sample={sorted(anom_pairs)[:20]}")
    print(f"  year unique={sorted(anom['year'].unique().tolist())}")
    print(
        f"  n_rows min={anom['n_rows'].min()} "
        f"median={int(anom['n_rows'].median())} "
        f"max={anom['n_rows'].max()}"
    )

    print("\n[OVERLAP]")
    print(f"  pair overlap={len(overlap)} {'OK' if len(overlap) == 0 else 'ERROR'}")

    if overlap:
        print(f"  overlapping pairs={sorted(overlap)[:20]}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q89 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q89.")
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
    for name in ['q89_normal', 'q89_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q89_normal"
    OUT_ANOM = args.output_dir / "q89_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(
        f"[CONFIG] Q89  "
        f"K={K}  "
        f"YEAR={NORMAL_YEAR}  "
        f"N_NORMAL={N_NORMAL}  "
        f"N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}"
    )
    print(f"[CONFIG] N_PAIRS={N_PAIRS}")
    print("[CONFIG] anomaly dimensions:")
    print("  - category-class pairs only")
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

    print("\n[GEN] s1_category_class")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_category_class",
            N_ANOMALY_PER_TYPE,
            sampler.s1_category_class_params,
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
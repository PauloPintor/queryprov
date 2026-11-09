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

# Só há uma dimensão anómala: reason.
# Portanto, usamos um número maior de anomalias deste tipo.
N_ANOMALY_PER_TYPE = 12_500

GENERATE_MIX = False

# K controla a concentração dos pesos dentro dos pools.
# K=0.5 -> distribuição mais concentrada nos reasons frequentes
# K=1.0 -> pesos proporcionais à frequência
# K>1   -> distribuição mais uniforme
K = 1.0

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.


# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
select
    ss_customer_sk,
    sum(act_sales) sumsales,
    array_agg(prov) as prov
from (
    select
        ss_item_sk,
        ss_ticket_number,
        ss_customer_sk,
        case
            when sr_return_quantity is not null then
                (ss_quantity - sr_return_quantity) * ss_sales_price
            else
                ss_quantity * ss_sales_price
        end act_sales,
        list_value(
            store_sales.prov,
            store_returns.prov,
            reason.prov
        ) as prov
    from store_sales
         left outer join store_returns
           on (sr_item_sk = ss_item_sk
               and sr_ticket_number = ss_ticket_number),
         reason
    where sr_reason_sk = r_reason_sk
      and r_reason_desc = '{reason}'
) t
group by ss_customer_sk
order by
    sumsales,
    ss_customer_sk
limit 100
"""


COUNT_TEMPLATE = """
select count(*) as n_rows
from (
    select
        ss_customer_sk
    from (
        select
            ss_customer_sk,
            case
                when sr_return_quantity is not null then
                    (ss_quantity - sr_return_quantity) * ss_sales_price
                else
                    ss_quantity * ss_sales_price
            end act_sales
        from store_sales
             left outer join store_returns
               on (sr_item_sk = ss_item_sk
                   and sr_ticket_number = ss_ticket_number),
             reason
        where sr_reason_sk = r_reason_sk
          and r_reason_desc = '{reason}'
    ) t
    group by ss_customer_sk
) g
"""


# ============================================================
# REASON FITTING
# ============================================================

def fit_reason_tiers(
    con: duckdb.DuckDBPyConnection,
    k: float,
    rng: np.random.Generator,
) -> dict:
    """
    Frequências dos reasons no espaço efetivo da query original.

    Mantém a forma original:
        store_sales LEFT OUTER JOIN store_returns ..., reason
        WHERE sr_reason_sk = r_reason_sk

    Como a condição sr_reason_sk = r_reason_sk elimina NULLs, o espaço efetivo
    são apenas linhas de store_sales com matching store_returns e reason.
    """

    df = con.execute("""
        SELECT
            r_reason_desc AS reason,
            COUNT(*) AS n_rows,
            COUNT(DISTINCT ss_customer_sk) AS n_customers
        FROM store_sales
             LEFT OUTER JOIN store_returns
               ON (sr_item_sk = ss_item_sk
                   AND sr_ticket_number = ss_ticket_number),
             reason
        WHERE sr_reason_sk = r_reason_sk
          AND r_reason_desc IS NOT NULL
        GROUP BY r_reason_desc
        ORDER BY n_rows DESC, r_reason_desc
    """).df()

    if len(df) < 4:
        raise ValueError(f"Poucos reasons distintos: {len(df)}")

    reasons = df["reason"].astype(str).tolist()
    counts = df["n_rows"].astype(float).to_numpy()

    n = len(reasons)

    # Como só há uma dimensão categórica, dividimos em dois pools disjuntos.
    # Com 35 reasons, isto dá 17 normais e 18 anómalos.
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

    normal_reasons = [
        reasons[i]
        for i in normal_idx
    ]

    anomaly_reasons = [
        reasons[i]
        for i in anomaly_idx
    ]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    alpha = 1.0 / max(k, 0.05)

    normal_w = normal_counts ** alpha
    normal_w = normal_w / normal_w.sum()

    anomaly_w = anomaly_counts ** alpha
    anomaly_w = anomaly_w / anomaly_w.sum()

    print("\n[REASON]")
    print(f"  n_reasons={n}  K={k}  alpha={alpha:.3f}")
    print(f"  normal n={len(normal_reasons)}  anomaly n={len(anomaly_reasons)}")
    print(f"  normal={normal_reasons}")
    print(f"  anomaly={anomaly_reasons}")

    print("\n[REASON FREQUENCIES]")
    print(df.to_string(index=False))

    return {
        "normal": normal_reasons,
        "normal_w": normal_w,
        "anomaly": anomaly_reasons,
        "anomaly_w": anomaly_w,
        "df": df,
    }


def sample_reason(
    pool: list[str],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> str:
    if not pool:
        raise ValueError("Pool de reasons vazio.")

    idx = int(
        rng.choice(
            len(pool),
            size=1,
            replace=True,
            p=weights,
        )[0]
    )

    return pool[idx]


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

        self.reason = fit_reason_tiers(
            con,
            K,
            rng,
        )

    def normal_params(self) -> dict:
        reason = sample_reason(
            self.reason["normal"],
            self.reason["normal_w"],
            self.rng,
        )

        return self._params(
            scenario="normal",
            label=0,
            reason=reason,
        )

    def s1_reason_params(self) -> dict:
        reason = sample_reason(
            self.reason["anomaly"],
            self.reason["anomaly_w"],
            self.rng,
        )

        return self._params(
            scenario="s1_reason",
            label=1,
            reason=reason,
        )

    def _params(
        self,
        scenario: str,
        label: int,
        reason: str,
    ) -> dict:
        return {
            "label": label,
            "scenario": scenario,
            "reason": reason,
            "provenance_semantics": "standard",
        }


# ============================================================
# RENDERING
# ============================================================

def _sql_quote_escape(s: str) -> str:
    return str(s).replace("'", "''")


def _prepare_sql_params(params: dict) -> dict:
    p = dict(params)
    p["reason"] = _sql_quote_escape(p["reason"])
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

    if param == "reason":
        norm_values = set(norm["reason"].astype(str))
        anom_values = set(g["reason"].astype(str))
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
    print(f"  reasons unique={norm['reason'].nunique()}")
    print(f"  reasons={sorted(norm['reason'].unique().tolist())}")
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
        print(f"  reasons unique={g['reason'].nunique()}")
        print(f"  reasons={sorted(g['reason'].unique().tolist())}")
        print(
            f"  n_rows min={g['n_rows'].min()} "
            f"median={int(g['n_rows'].median())} "
            f"max={g['n_rows'].max()}"
        )

        if scenario == "s1_reason":
            print_overlap_for_param(
                norm,
                anom,
                scenario,
                "reason",
                should_be_disjoint=True,
            )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q93 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q93.")
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
    for name in ['q93_normal', 'q93_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q93_normal"
    OUT_ANOM = args.output_dir / "q93_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(
        f"[CONFIG] Q93  "
        f"K={K}  "
        f"N_NORMAL={N_NORMAL}  "
        f"N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}"
    )
    print("[CONFIG] anomaly dimensions:")
    print("  - return reason description")
    print("[CONFIG] provenance semantics:")
    print("  - standard why-provenance")
    print("  - witness = store_sales.prov · store_returns.prov · reason.prov")
    print("[CONFIG] query structure:")
    print("  - original TPC-DS Q93 structure preserved")
    print("  - only provenance column added")
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

    print("\n[GEN] s1_reason")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_reason",
            N_ANOMALY_PER_TYPE,
            sampler.s1_reason_params,
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
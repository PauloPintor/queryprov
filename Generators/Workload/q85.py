#!/usr/bin/env python3
from __future__ import annotations

import argparse

import csv
import os
from pathlib import Path
from typing import Any

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

# K=0.5 -> sampling mais concentrado nos valores frequentes
# K=1.0 -> sampling empírico
# K>1   -> sampling mais uniforme
K = 1.0

# Mantemos igual ao Q48 para reduzir variabilidade.
YEAR = 2002

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.


# ============================================================
# SQL TEMPLATE — Q85 WITH PROVENANCE
# ============================================================

SQL_TEMPLATE = """
select
    substr(r_reason_desc, 1, 20) as reason_desc,
    avg(ws_quantity) as avg_ws_quantity,
    avg(wr_refunded_cash) as avg_wr_refunded_cash,
    avg(wr_fee) as avg_wr_fee,
    array_agg(
        list_value(
            web_sales.prov,
            web_returns.prov,
            web_page.prov,
            cd1.prov,
            cd2.prov,
            customer_address.prov,
            date_dim.prov,
            reason.prov
        )
    ) as prov
from web_sales,
     web_returns,
     web_page,
     customer_demographics cd1,
     customer_demographics cd2,
     customer_address,
     date_dim,
     reason
where ws_web_page_sk = wp_web_page_sk
  and ws_item_sk = wr_item_sk
  and ws_order_number = wr_order_number
  and ws_sold_date_sk = d_date_sk
  and d_year = {year}
  and cd1.cd_demo_sk = wr_refunded_cdemo_sk
  and cd2.cd_demo_sk = wr_returning_cdemo_sk
  and ca_address_sk = wr_refunded_addr_sk
  and r_reason_sk = wr_reason_sk

  and (
      (
          cd1.cd_marital_status = '{ms1}'
          and cd1.cd_marital_status = cd2.cd_marital_status
          and cd1.cd_education_status = '{es1}'
          and cd1.cd_education_status = cd2.cd_education_status
          and ws_sales_price between 100.00 and 150.00
      )
   or (
          cd1.cd_marital_status = '{ms2}'
          and cd1.cd_marital_status = cd2.cd_marital_status
          and cd1.cd_education_status = '{es2}'
          and cd1.cd_education_status = cd2.cd_education_status
          and ws_sales_price between 50.00 and 100.00
      )
   or (
          cd1.cd_marital_status = '{ms3}'
          and cd1.cd_marital_status = cd2.cd_marital_status
          and cd1.cd_education_status = '{es3}'
          and cd1.cd_education_status = cd2.cd_education_status
          and ws_sales_price between 150.00 and 200.00
      )
  )

  and (
      (
          ca_country = 'United States'
          and ca_state in ('{s1}', '{s2}', '{s3}')
          and ws_net_profit between 100 and 200
      )
   or (
          ca_country = 'United States'
          and ca_state in ('{s4}', '{s5}', '{s6}')
          and ws_net_profit between 150 and 300
      )
   or (
          ca_country = 'United States'
          and ca_state in ('{s7}', '{s8}', '{s9}')
          and ws_net_profit between 50 and 250
      )
  )
group by r_reason_desc
order by
    substr(r_reason_desc, 1, 20),
    avg(ws_quantity),
    avg(wr_refunded_cash),
    avg(wr_fee)
limit 100
"""


COUNT_TEMPLATE = """
select count(*) as n_rows
from web_sales,
     web_returns,
     web_page,
     customer_demographics cd1,
     customer_demographics cd2,
     customer_address,
     date_dim,
     reason
where ws_web_page_sk = wp_web_page_sk
  and ws_item_sk = wr_item_sk
  and ws_order_number = wr_order_number
  and ws_sold_date_sk = d_date_sk
  and d_year = {year}
  and cd1.cd_demo_sk = wr_refunded_cdemo_sk
  and cd2.cd_demo_sk = wr_returning_cdemo_sk
  and ca_address_sk = wr_refunded_addr_sk
  and r_reason_sk = wr_reason_sk

  and (
      (
          cd1.cd_marital_status = '{ms1}'
          and cd1.cd_marital_status = cd2.cd_marital_status
          and cd1.cd_education_status = '{es1}'
          and cd1.cd_education_status = cd2.cd_education_status
          and ws_sales_price between 100.00 and 150.00
      )
   or (
          cd1.cd_marital_status = '{ms2}'
          and cd1.cd_marital_status = cd2.cd_marital_status
          and cd1.cd_education_status = '{es2}'
          and cd1.cd_education_status = cd2.cd_education_status
          and ws_sales_price between 50.00 and 100.00
      )
   or (
          cd1.cd_marital_status = '{ms3}'
          and cd1.cd_marital_status = cd2.cd_marital_status
          and cd1.cd_education_status = '{es3}'
          and cd1.cd_education_status = cd2.cd_education_status
          and ws_sales_price between 150.00 and 200.00
      )
  )

  and (
      (
          ca_country = 'United States'
          and ca_state in ('{s1}', '{s2}', '{s3}')
          and ws_net_profit between 100 and 200
      )
   or (
          ca_country = 'United States'
          and ca_state in ('{s4}', '{s5}', '{s6}')
          and ws_net_profit between 150 and 300
      )
   or (
          ca_country = 'United States'
          and ca_state in ('{s7}', '{s8}', '{s9}')
          and ws_net_profit between 50 and 250
      )
  )
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


def weighted_sample_without_replacement(
    pool: list,
    weights: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> list:
    if len(pool) < n:
        raise ValueError(f"Pool too small: {len(pool)} < {n}")

    idx = rng.choice(
        len(pool),
        size=n,
        replace=False,
        p=weights,
    )

    return [pool[int(i)] for i in idx]


def assert_paths() -> None:
    db_path = Path(DB_PATH)

    if not db_path.exists():
        raise FileNotFoundError(
            f"DUCKDB_PATH não existe: {db_path}\n"
            f"Define com:\n"
            f'  DUCKDB_PATH="/path/to/tpcds_prov.duckdb" python3 generate_q85_queries.py'
        )

    EXTERNAL_BASE_DIR.mkdir(parents=True, exist_ok=True)

    test_file = EXTERNAL_BASE_DIR / "_write_test.tmp"

    try:
        test_file.write_text("ok\n", encoding="utf-8")
        test_file.unlink()
    except Exception as exc:
        raise RuntimeError(
            f"Não consigo escrever em {EXTERNAL_BASE_DIR}. "
            f"Check that the selected output directory is writable."
        ) from exc


# ============================================================
# DEMOGRAPHIC TIERS
# ============================================================

def fit_demographic_tiers(
    con: duckdb.DuckDBPyConnection,
    k: float,
    rng: np.random.Generator,
) -> dict:
    """
    Perfil demográfico composto:
      (cd_marital_status, cd_education_status)

    Em Q85, cd1 e cd2 têm de ter o mesmo marital_status e education_status.
    As estatísticas são estimadas no join path da própria Q85.
    """

    df = con.execute(f"""
        SELECT
            cd1.cd_marital_status AS ms,
            cd1.cd_education_status AS es,
            COUNT(*) AS n_rows
        FROM web_sales ws
        JOIN web_returns wr
          ON ws.ws_item_sk = wr.wr_item_sk
         AND ws.ws_order_number = wr.wr_order_number
        JOIN customer_demographics cd1
          ON cd1.cd_demo_sk = wr.wr_refunded_cdemo_sk
        JOIN customer_demographics cd2
          ON cd2.cd_demo_sk = wr.wr_returning_cdemo_sk
        JOIN date_dim d
          ON ws.ws_sold_date_sk = d.d_date_sk
        WHERE d.d_year = {YEAR}
          AND cd1.cd_marital_status = cd2.cd_marital_status
          AND cd1.cd_education_status = cd2.cd_education_status
          AND cd1.cd_marital_status IS NOT NULL
          AND cd1.cd_education_status IS NOT NULL
        GROUP BY
            cd1.cd_marital_status,
            cd1.cd_education_status
        ORDER BY n_rows DESC,
                 ms,
                 es
    """).df()

    if len(df) < 6:
        raise ValueError(
            f"Poucos pares demográficos distintos: {len(df)}. "
            "A Q85 precisa de pelo menos 3 normais e 3 anómalos."
        )

    pairs = [
        (
            str(r.ms),
            str(r.es),
        )
        for r in df.itertuples(index=False)
    ]

    counts = df["n_rows"].astype(float).to_numpy()
    n = len(pairs)

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

    normal_pairs = [pairs[i] for i in normal_idx]
    anomaly_pairs = [pairs[i] for i in anomaly_idx]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    normal_w = _weights_from_counts(normal_counts, k)
    anomaly_w = _weights_from_counts(anomaly_counts, k)

    print("\n[DEMOGRAPHIC PROFILE]")
    print(f"  year={YEAR}")
    print(f"  n_pairs={n}  K={k}")
    print(f"  normal n={len(normal_pairs)}  anomaly n={len(anomaly_pairs)}")
    print(f"  normal={normal_pairs}")
    print(f"  anomaly={anomaly_pairs}")

    return {
        "normal": normal_pairs,
        "normal_w": normal_w,
        "anomaly": anomaly_pairs,
        "anomaly_w": anomaly_w,
        "df": df,
    }


# ============================================================
# STATE TIERS
# ============================================================

def fit_state_tiers(
    con: duckdb.DuckDBPyConnection,
    k: float,
    rng: np.random.Generator,
) -> dict:
    """
    Estados são domínio categórico.

    A Q85 usa nove estados por query, logo cada pool precisa de >= 9.
    As estatísticas são estimadas no join path web_sales/web_returns/address.
    """

    df = con.execute(f"""
        SELECT
            ca.ca_state AS state,
            COUNT(*) AS n_rows
        FROM web_sales ws
        JOIN web_returns wr
          ON ws.ws_item_sk = wr.wr_item_sk
         AND ws.ws_order_number = wr.wr_order_number
        JOIN customer_address ca
          ON ca.ca_address_sk = wr.wr_refunded_addr_sk
        JOIN date_dim d
          ON ws.ws_sold_date_sk = d.d_date_sk
        WHERE d.d_year = {YEAR}
          AND ca.ca_country = 'United States'
          AND ca.ca_state IS NOT NULL
        GROUP BY ca.ca_state
        ORDER BY n_rows DESC,
                 state
    """).df()

    if len(df) < 18:
        raise ValueError(
            f"Poucos estados distintos: {len(df)}. "
            "A Q85 precisa de pelo menos 9 estados normais e 9 anómalos."
        )

    states = df["state"].astype(str).tolist()
    counts = df["n_rows"].astype(float).to_numpy()
    n = len(states)

    normal_size = n // 2
    normal_size = max(9, min(normal_size, n - 9))

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

    normal_states = [states[i] for i in normal_idx]
    anomaly_states = [states[i] for i in anomaly_idx]

    normal_counts = counts[normal_idx]
    anomaly_counts = counts[anomaly_idx]

    normal_w = _weights_from_counts(normal_counts, k)
    anomaly_w = _weights_from_counts(anomaly_counts, k)

    print("\n[STATE]")
    print(f"  year={YEAR}")
    print(f"  n_states={n}  K={k}")
    print(f"  normal n={len(normal_states)}  anomaly n={len(anomaly_states)}")
    print(f"  normal={normal_states}")
    print(f"  anomaly={anomaly_states}")

    return {
        "normal": normal_states,
        "normal_w": normal_w,
        "anomaly": anomaly_states,
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
        self.demographic = fit_demographic_tiers(con, K, rng)
        self.state = fit_state_tiers(con, K, rng)

    def normal_params(self) -> dict:
        demo = weighted_sample_without_replacement(
            self.demographic["normal"],
            self.demographic["normal_w"],
            self.rng,
            3,
        )

        states = weighted_sample_without_replacement(
            self.state["normal"],
            self.state["normal_w"],
            self.rng,
            9,
        )

        return self._build(
            scenario="normal",
            label=0,
            demo=demo,
            states=states,
        )

    def s1_demographic_params(self) -> dict:
        demo = weighted_sample_without_replacement(
            self.demographic["anomaly"],
            self.demographic["anomaly_w"],
            self.rng,
            3,
        )

        states = weighted_sample_without_replacement(
            self.state["normal"],
            self.state["normal_w"],
            self.rng,
            9,
        )

        return self._build(
            scenario="s1_demographic",
            label=1,
            demo=demo,
            states=states,
        )

    def s1_state_params(self) -> dict:
        demo = weighted_sample_without_replacement(
            self.demographic["normal"],
            self.demographic["normal_w"],
            self.rng,
            3,
        )

        states = weighted_sample_without_replacement(
            self.state["anomaly"],
            self.state["anomaly_w"],
            self.rng,
            9,
        )

        return self._build(
            scenario="s1_state",
            label=1,
            demo=demo,
            states=states,
        )

    def s1_mix_params(self) -> dict:
        demo = weighted_sample_without_replacement(
            self.demographic["anomaly"],
            self.demographic["anomaly_w"],
            self.rng,
            3,
        )

        states = weighted_sample_without_replacement(
            self.state["anomaly"],
            self.state["anomaly_w"],
            self.rng,
            9,
        )

        return self._build(
            scenario="s1_mix",
            label=1,
            demo=demo,
            states=states,
        )

    def _build(
        self,
        scenario: str,
        label: int,
        demo: list[tuple[str, str]],
        states: list[str],
    ) -> dict:
        return {
            "scenario": scenario,
            "label": label,
            "year": YEAR,

            "ms1": demo[0][0],
            "es1": demo[0][1],
            "ms2": demo[1][0],
            "es2": demo[1][1],
            "ms3": demo[2][0],
            "es3": demo[2][1],

            "s1": states[0],
            "s2": states[1],
            "s3": states[2],
            "s4": states[3],
            "s5": states[4],
            "s6": states[5],
            "s7": states[6],
            "s8": states[7],
            "s9": states[8],
        }


# ============================================================
# RENDERING
# ============================================================

def _prepare_sql_params(params: dict) -> dict:
    p = dict(params)

    for key in [
        "ms1", "es1",
        "ms2", "es2",
        "ms3", "es3",
        "s1", "s2", "s3",
        "s4", "s5", "s6",
        "s7", "s8", "s9",
    ]:
        p[key] = sql_escape(p[key])

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
            n_rows = int(con.execute(render_count(params)).fetchone()[0])
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
                "n_rows": n_rows,
                **params,
            }
        )

        i += 1

        if i % 1000 == 0:
            print(f"  [GEN] {prefix}: {i}/{n}")

        if attempts % 10_000 == 0:
            print(
                f"  [ATTEMPTS] {prefix}: generated={i}/{n} attempts={attempts}",
                flush=True,
            )

    write_metadata(
        out_dir / "metadata.csv",
        rows,
    )

    print(f"  [OK] {prefix}: {len(rows)} queries, attempts={attempts}")

    return rows


# ============================================================
# DIAGNOSTICS
# ============================================================

def _demo_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    return (
        set(zip(df["ms1"].astype(str), df["es1"].astype(str)))
        | set(zip(df["ms2"].astype(str), df["es2"].astype(str)))
        | set(zip(df["ms3"].astype(str), df["es3"].astype(str)))
    )


def _state_set(df: pd.DataFrame) -> set[str]:
    out: set[str] = set()

    for c in ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]:
        out |= set(df[c].astype(str))

    return out


def print_diagnostics(
    normal_rows: list[dict],
    anom_rows: list[dict],
) -> None:
    norm = pd.DataFrame(normal_rows)
    anom = pd.DataFrame(anom_rows)

    norm_demo = _demo_set(norm)
    norm_states = _state_set(norm)

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    print(f"\n[NORMAL] n={len(norm)}")
    print(f"  demographic pairs unique={len(norm_demo)}")
    print(f"  states unique={len(norm_states)}")
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

        g_demo = _demo_set(g)
        g_states = _state_set(g)

        demo_overlap = norm_demo & g_demo
        state_overlap = norm_states & g_states

        if scenario == "s1_demographic":
            demo_ok = len(demo_overlap) == 0
            state_ok = True
        elif scenario == "s1_state":
            demo_ok = True
            state_ok = len(state_overlap) == 0
        else:
            demo_ok = len(demo_overlap) == 0
            state_ok = len(state_overlap) == 0

        print(f"\n[{scenario}] n={len(g)}")
        print(f"  demographic pairs unique={len(g_demo)}")
        print(f"  states unique={len(g_states)}")
        print(f"  year unique={sorted(g['year'].unique().tolist())}")
        print(
            f"  n_rows min={g['n_rows'].min()} "
            f"median={int(g['n_rows'].median())} "
            f"max={g['n_rows'].max()}"
        )
        print(f"  demographic overlap={len(demo_overlap)} {'OK' if demo_ok else 'ERROR'}")
        print(f"  state overlap={len(state_overlap)} {'OK' if state_ok else 'ERROR'}")

        if not demo_ok:
            print(f"    overlapping demographic pairs={sorted(demo_overlap)[:20]}")

        if not state_ok:
            print(f"    overlapping states={sorted(state_overlap)[:20]}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q85 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q85.")
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
    for name in ['q85_normal', 'q85_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q85_normal"
    OUT_ANOM = args.output_dir / "q85_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(
        f"[CONFIG] Q85  "
        f"K={K}  "
        f"N_NORMAL={N_NORMAL}  "
        f"N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}"
    )
    print(f"[CONFIG] fixed year={YEAR}")
    print("[CONFIG] anomaly dimensions:")
    print("  - joint demographic profile: marital_status + education_status")
    print("  - customer states")
    print(f"[CONFIG] generate_mix={GENERATE_MIX}")
    print(f"[OUT] normal={OUT_NORMAL}")
    print(f"[OUT] anomalies={OUT_ANOM}")

    assert_paths()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET search_path = ?", [DB_SCHEMA])

    rng = np.random.default_rng(SEED)

    try:
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

        print("\n[GEN] s1_demographic")
        anom_rows.extend(
            write_queries(
                OUT_ANOM,
                "s1_demographic",
                N_ANOMALY_PER_TYPE,
                sampler.s1_demographic_params,
                con,
            )
        )

        print("\n[GEN] s1_state")
        anom_rows.extend(
            write_queries(
                OUT_ANOM,
                "s1_state",
                N_ANOMALY_PER_TYPE,
                sampler.s1_state_params,
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

    finally:
        con.close()

    print_diagnostics(normal_rows, anom_rows)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
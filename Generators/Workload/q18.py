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

# K=0.5 -> distribuição mais concentrada
# K=1.0 -> distribuição empírica
# K>1   -> mais uniforme
K = 1.0

YEAR = 2002

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.


# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
select
       i_item_id,
       ca_country,
       ca_state,
       ca_county,
       avg(cast(cs_quantity as decimal(12,2))) as agg1,
       avg(cast(cs_list_price as decimal(12,2))) as agg2,
       avg(cast(cs_coupon_amt as decimal(12,2))) as agg3,
       avg(cast(cs_sales_price as decimal(12,2))) as agg4,
       avg(cast(cs_net_profit as decimal(12,2))) as agg5,
       avg(cast(c_birth_year as decimal(12,2))) as agg6,
       avg(cast(cd1.cd_dep_count as decimal(12,2))) as agg7,
       array_agg(
           list_value(
               catalog_sales.prov,
               cd1.prov,
               cd2.prov,
               customer.prov,
               customer_address.prov,
               date_dim.prov,
               item.prov
           )
       ) as prov
from catalog_sales,
     customer_demographics cd1,
     customer_demographics cd2,
     customer,
     customer_address,
     date_dim,
     item
where cs_sold_date_sk = d_date_sk
  and cs_item_sk = i_item_sk
  and cs_bill_cdemo_sk = cd1.cd_demo_sk
  and cs_bill_customer_sk = c_customer_sk

  and cd1.cd_gender = '{gen}'
  and cd1.cd_education_status = '{es}'

  and c_current_cdemo_sk = cd2.cd_demo_sk
  and c_current_addr_sk = ca_address_sk

  and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})

  and d_year = {year}

  and ca_state in ('{s1}', '{s2}', '{s3}', '{s4}', '{s5}', '{s6}', '{s7}')

group by rollup (
    i_item_id,
    ca_country,
    ca_state,
    ca_county
)

order by
    ca_country,
    ca_state,
    ca_county,
    i_item_id
"""


COUNT_TEMPLATE = """
select count(*) from (
    select
           i_item_id,
           ca_country,
           ca_state,
           ca_county
    from catalog_sales,
         customer_demographics cd1,
         customer_demographics cd2,
         customer,
         customer_address,
         date_dim,
         item
    where cs_sold_date_sk = d_date_sk
      and cs_item_sk = i_item_sk
      and cs_bill_cdemo_sk = cd1.cd_demo_sk
      and cs_bill_customer_sk = c_customer_sk

      and cd1.cd_gender = '{gen}'
      and cd1.cd_education_status = '{es}'

      and c_current_cdemo_sk = cd2.cd_demo_sk
      and c_current_addr_sk = ca_address_sk

      and c_birth_month in ({m1}, {m2}, {m3}, {m4}, {m5}, {m6})

      and d_year = {year}

      and ca_state in ('{s1}', '{s2}', '{s3}', '{s4}', '{s5}', '{s6}', '{s7}')

    group by rollup (
        i_item_id,
        ca_country,
        ca_state,
        ca_county
    )
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


def sample_one(
    pool: list,
    weights: np.ndarray,
    rng: np.random.Generator,
):
    idx = int(
        rng.choice(
            len(pool),
            size=1,
            replace=False,
            p=weights,
        )[0]
    )
    return pool[idx]


def sample_without_replacement_weighted(
    pool: list,
    weights: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> list:
    if len(pool) < n:
        raise ValueError(f"Pool tem {len(pool)} valores, mas precisa de {n}.")

    idx = rng.choice(
        len(pool),
        size=n,
        replace=False,
        p=weights,
    )

    return [pool[int(i)] for i in idx]


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
      (cd_gender, cd_education_status)

    Nenhum atributo é tratado como anómalo isoladamente.
    A anomalia é definida sobre o par categórico usado no predicado.
    """

    df = con.execute(f"""
        SELECT
            cd.cd_gender AS gen,
            cd.cd_education_status AS es,
            COUNT(*) AS n_sales
        FROM catalog_sales cs
        JOIN customer_demographics cd
          ON cs.cs_bill_cdemo_sk = cd.cd_demo_sk
        JOIN date_dim d
          ON cs.cs_sold_date_sk = d.d_date_sk
        WHERE d.d_year = {YEAR}
          AND cd.cd_gender IS NOT NULL
          AND cd.cd_education_status IS NOT NULL
        GROUP BY
            cd.cd_gender,
            cd.cd_education_status
        ORDER BY n_sales DESC,
                 gen,
                 es
    """).df()

    if len(df) < 2:
        raise ValueError(f"Poucos pares demográficos distintos: {len(df)}")

    pairs = [
        (
            str(r.gen),
            str(r.es),
        )
        for r in df.itertuples(index=False)
    ]

    counts = df["n_sales"].astype(float).to_numpy()
    n = len(pairs)

    # Como escolhemos 1 perfil por query, basta garantir >=1 em cada pool.
    # Mantemos divisão equilibrada, como nos geradores novos.
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

    A query usa ca_state através do caminho:
      catalog_sales -> customer -> customer_address

    Como cada query precisa de 7 estados, ambos os pools têm de ter >= 7.
    """

    df = con.execute(f"""
        SELECT
            ca.ca_state AS state,
            COUNT(*) AS n_sales
        FROM catalog_sales cs
        JOIN customer c
          ON cs.cs_bill_customer_sk = c.c_customer_sk
        JOIN customer_address ca
          ON c.c_current_addr_sk = ca.ca_address_sk
        JOIN date_dim d
          ON cs.cs_sold_date_sk = d.d_date_sk
        WHERE d.d_year = {YEAR}
          AND ca.ca_state IS NOT NULL
        GROUP BY ca.ca_state
        ORDER BY n_sales DESC,
                 state
    """).df()

    if len(df) < 14:
        raise ValueError(
            f"Poucos estados distintos: {len(df)}. "
            "A Q18 precisa de pelo menos 7 estados normais e 7 anómalos."
        )

    states = df["state"].astype(str).tolist()
    counts = df["n_sales"].astype(float).to_numpy()
    n = len(states)

    normal_size = n // 2
    normal_size = max(7, min(normal_size, n - 7))

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

    def random_months(self) -> list[int]:
        months = self.rng.choice(
            np.arange(1, 13),
            size=6,
            replace=False,
        )

        return sorted(int(x) for x in months)

    def normal_params(self) -> dict:
        demo = sample_one(
            self.demographic["normal"],
            self.demographic["normal_w"],
            self.rng,
        )

        states = sample_without_replacement_weighted(
            self.state["normal"],
            self.state["normal_w"],
            self.rng,
            7,
        )

        return self._build(
            scenario="normal",
            label=0,
            demo=demo,
            states=states,
        )

    def s1_demographic_params(self) -> dict:
        demo = sample_one(
            self.demographic["anomaly"],
            self.demographic["anomaly_w"],
            self.rng,
        )

        states = sample_without_replacement_weighted(
            self.state["normal"],
            self.state["normal_w"],
            self.rng,
            7,
        )

        return self._build(
            scenario="s1_demographic",
            label=1,
            demo=demo,
            states=states,
        )

    def s1_state_params(self) -> dict:
        demo = sample_one(
            self.demographic["normal"],
            self.demographic["normal_w"],
            self.rng,
        )

        states = sample_without_replacement_weighted(
            self.state["anomaly"],
            self.state["anomaly_w"],
            self.rng,
            7,
        )

        return self._build(
            scenario="s1_state",
            label=1,
            demo=demo,
            states=states,
        )

    def s1_mix_params(self) -> dict:
        demo = sample_one(
            self.demographic["anomaly"],
            self.demographic["anomaly_w"],
            self.rng,
        )

        states = sample_without_replacement_weighted(
            self.state["anomaly"],
            self.state["anomaly_w"],
            self.rng,
            7,
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
        demo: tuple[str, str],
        states: list[str],
    ) -> dict:
        months = self.random_months()

        gen, es = demo

        return {
            "scenario": scenario,
            "label": label,
            "year": YEAR,
            "gen": gen,
            "es": es,
            "m1": months[0],
            "m2": months[1],
            "m3": months[2],
            "m4": months[3],
            "m5": months[4],
            "m6": months[5],
            "s1": states[0],
            "s2": states[1],
            "s3": states[2],
            "s4": states[3],
            "s5": states[4],
            "s6": states[5],
            "s7": states[6],
        }


# ============================================================
# RENDERING
# ============================================================

def _prepare_sql_params(params: dict) -> dict:
    p = dict(params)

    for key in [
        "gen",
        "es",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
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
    seen = set()

    max_attempts = max(100_000, n * 50)

    while i < n:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Too many attempts while generating {prefix}: "
                f"{attempts} attempts for {i}/{n} queries. "
                "The parameter space may be too small for unique generation."
            )

        params = sampler_fn()

        # Metadata signature; enough to avoid exact duplicates.
        signature = (
            params["scenario"],
            params["gen"],
            params["es"],
            params["year"],
            params["m1"],
            params["m2"],
            params["m3"],
            params["m4"],
            params["m5"],
            params["m6"],
            tuple(sorted([
                params["s1"],
                params["s2"],
                params["s3"],
                params["s4"],
                params["s5"],
                params["s6"],
                params["s7"],
            ])),
        )

        if signature in seen:
            continue

        seen.add(signature)

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
                "n_rows": n_rows,
                **params,
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

def _demo_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            df["gen"].astype(str),
            df["es"].astype(str),
        )
    )


def _state_set(df: pd.DataFrame) -> set[str]:
    return (
        set(df["s1"].astype(str))
        | set(df["s2"].astype(str))
        | set(df["s3"].astype(str))
        | set(df["s4"].astype(str))
        | set(df["s5"].astype(str))
        | set(df["s6"].astype(str))
        | set(df["s7"].astype(str))
    )


def print_diagnostics(normal_rows: list[dict], anom_rows: list[dict]) -> None:
    norm = pd.DataFrame(normal_rows)
    anom = pd.DataFrame(anom_rows)

    norm_demo = _demo_set(norm)
    norm_states = _state_set(norm)

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    print(f"\n[NORMAL] n={len(norm)}")
    print(f"  demographic profiles unique={len(norm_demo)}")
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
        print(f"  demographic profiles unique={len(g_demo)}")
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
            print(f"    overlapping demographic profiles={sorted(demo_overlap)}")

        if not state_ok:
            print(f"    overlapping states={sorted(state_overlap)}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q18 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q18.")
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
    for name in ['q18_normal', 'q18_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q18_normal"
    OUT_ANOM = args.output_dir / "q18_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(
        f"[CONFIG] Q18  "
        f"K={K}  "
        f"N_NORMAL={N_NORMAL}  "
        f"N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}"
    )
    print(f"[CONFIG] fixed year={YEAR}")
    print("[CONFIG] anomaly dimensions:")
    print("  - joint demographic profile: gender + education_status")
    print("  - customer states")
    print("[CONFIG] birth months = uniform, not anomaly")
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

    print("\n[GEN] s1_demographic")
    anom_all.extend(
        write_queries(
            OUT_ANOM,
            "s1_demographic",
            N_ANOMALY_PER_TYPE,
            sampler.s1_demographic_params,
            con,
        )
    )

    print("\n[GEN] s1_state")
    anom_all.extend(
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
        anom_all.extend(
            write_queries(
                OUT_ANOM,
                "s1_mix",
                N_ANOMALY_PER_TYPE,
                sampler.s1_mix_params,
                con,
            )
        )

    write_metadata(OUT_ANOM / "metadata.csv", anom_all)

    con.close()

    print_diagnostics(normal_rows, anom_all)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
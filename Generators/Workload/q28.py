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

# K=0.5 -> normal mais compacto
# K=1.0 -> normal mais largo
K = 1.0

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.

# Parâmetros contínuos do template
PARAMS = {
    "listprice": {
        "lo": 0,
        "hi": 190,
        "window": 10,
    },
    "couponamt": {
        "lo": 0,
        "hi": 18_000,
        "window": 1_000,
    },
    "wholesalecost": {
        "lo": 0,
        "hi": 80,
        "window": 20,
    },
}

LP_WINDOW = PARAMS["listprice"]["window"]
CA_WINDOW = PARAMS["couponamt"]["window"]
WC_WINDOW = PARAMS["wholesalecost"]["window"]


# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
select *,
       list_value(B1.prov, B2.prov, B3.prov, B4.prov, B5.prov, B6.prov) as prov
from
     (select avg(ss_list_price) B1_LP
            ,count(ss_list_price) B1_CNT
            ,count(distinct ss_list_price) B1_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 0 and 5
        and (ss_list_price between {lp1} and {lp1}+{lp_window}
             or ss_coupon_amt between {ca1} and {ca1}+{ca_window}
             or ss_wholesale_cost between {wc1} and {wc1}+{wc_window})) B1,
     (select avg(ss_list_price) B2_LP
            ,count(ss_list_price) B2_CNT
            ,count(distinct ss_list_price) B2_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 6 and 10
        and (ss_list_price between {lp2} and {lp2}+{lp_window}
          or ss_coupon_amt between {ca2} and {ca2}+{ca_window}
          or ss_wholesale_cost between {wc2} and {wc2}+{wc_window})) B2,
     (select avg(ss_list_price) B3_LP
            ,count(ss_list_price) B3_CNT
            ,count(distinct ss_list_price) B3_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 11 and 15
        and (ss_list_price between {lp3} and {lp3}+{lp_window}
          or ss_coupon_amt between {ca3} and {ca3}+{ca_window}
          or ss_wholesale_cost between {wc3} and {wc3}+{wc_window})) B3,
     (select avg(ss_list_price) B4_LP
            ,count(ss_list_price) B4_CNT
            ,count(distinct ss_list_price) B4_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 16 and 20
        and (ss_list_price between {lp4} and {lp4}+{lp_window}
          or ss_coupon_amt between {ca4} and {ca4}+{ca_window}
          or ss_wholesale_cost between {wc4} and {wc4}+{wc_window})) B4,
     (select avg(ss_list_price) B5_LP
            ,count(ss_list_price) B5_CNT
            ,count(distinct ss_list_price) B5_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 21 and 25
        and (ss_list_price between {lp5} and {lp5}+{lp_window}
          or ss_coupon_amt between {ca5} and {ca5}+{ca_window}
          or ss_wholesale_cost between {wc5} and {wc5}+{wc_window})) B5,
     (select avg(ss_list_price) B6_LP
            ,count(ss_list_price) B6_CNT
            ,count(distinct ss_list_price) B6_CNTD
            ,array_agg(store_sales.prov) prov
      from store_sales
      where ss_quantity between 26 and 30
        and (ss_list_price between {lp6} and {lp6}+{lp_window}
          or ss_coupon_amt between {ca6} and {ca6}+{ca_window}
          or ss_wholesale_cost between {wc6} and {wc6}+{wc_window})) B6
limit 100
"""


VALIDATION_TEMPLATE = """
select
    B1_CNT,
    B2_CNT,
    B3_CNT,
    B4_CNT,
    B5_CNT,
    B6_CNT
from
     (select count(ss_list_price) B1_CNT
      from store_sales
      where ss_quantity between 0 and 5
        and (ss_list_price between {lp1} and {lp1}+{lp_window}
             or ss_coupon_amt between {ca1} and {ca1}+{ca_window}
             or ss_wholesale_cost between {wc1} and {wc1}+{wc_window})) B1,
     (select count(ss_list_price) B2_CNT
      from store_sales
      where ss_quantity between 6 and 10
        and (ss_list_price between {lp2} and {lp2}+{lp_window}
          or ss_coupon_amt between {ca2} and {ca2}+{ca_window}
          or ss_wholesale_cost between {wc2} and {wc2}+{wc_window})) B2,
     (select count(ss_list_price) B3_CNT
      from store_sales
      where ss_quantity between 11 and 15
        and (ss_list_price between {lp3} and {lp3}+{lp_window}
          or ss_coupon_amt between {ca3} and {ca3}+{ca_window}
          or ss_wholesale_cost between {wc3} and {wc3}+{wc_window})) B3,
     (select count(ss_list_price) B4_CNT
      from store_sales
      where ss_quantity between 16 and 20
        and (ss_list_price between {lp4} and {lp4}+{lp_window}
          or ss_coupon_amt between {ca4} and {ca4}+{ca_window}
          or ss_wholesale_cost between {wc4} and {wc4}+{wc_window})) B4,
     (select count(ss_list_price) B5_CNT
      from store_sales
      where ss_quantity between 21 and 25
        and (ss_list_price between {lp5} and {lp5}+{lp_window}
          or ss_coupon_amt between {ca5} and {ca5}+{ca_window}
          or ss_wholesale_cost between {wc5} and {wc5}+{wc_window})) B5,
     (select count(ss_list_price) B6_CNT
      from store_sales
      where ss_quantity between 26 and 30
        and (ss_list_price between {lp6} and {lp6}+{lp_window}
          or ss_coupon_amt between {ca6} and {ca6}+{ca_window}
          or ss_wholesale_cost between {wc6} and {wc6}+{wc_window})) B6
"""


# ============================================================
# RANGES
# ============================================================

def compute_range(lo: int, hi: int, k: float) -> dict:
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


def print_ranges(ranges: dict) -> None:
    print("\n[RANGES]")
    for name, r in ranges.items():
        print(f"\n[{name}]")
        print(f"  domain=[{r['lo']}, {r['hi']}]")
        print(f"  mu={r['mu']:.2f} sigma={r['sigma']:.2f}")
        print(f"  normal=[{r['lo_normal']}, {r['hi_normal']}]")
        parts = []
        if r["has_left"]:
            parts.append(f"[{r['lo_anom_left']}, {r['hi_anom_left']}]")
        if r["has_right"]:
            parts.append(f"[{r['lo_anom_right']}, {r['hi_anom_right']}]")
        print(f"  anomaly={parts}")


# ============================================================
# SAMPLING
# ============================================================

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


def sample_normal_value(r: dict, rng: np.random.Generator) -> int:
    # sigma interno compacto: o intervalo normal representa aproximadamente 6σ.
    sigma_eff = max((r["hi_normal"] - r["lo_normal"]) / 6.0, 1e-6)

    return sample_truncated_normal(
        r["mu"],
        sigma_eff,
        r["lo_normal"],
        r["hi_normal"],
        rng,
    )


def sample_anomaly_value(r: dict, rng: np.random.Generator) -> int:
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


def sample_six(r: dict, mode: str, rng: np.random.Generator) -> list[int]:
    if mode == "normal":
        return [sample_normal_value(r, rng) for _ in range(6)]

    if mode == "anomaly":
        return [sample_anomaly_value(r, rng) for _ in range(6)]

    raise ValueError(f"Modo inválido: {mode}")


# ============================================================
# SAMPLER
# ============================================================

class QuerySampler:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.ranges = {
            name: compute_range(cfg["lo"], cfg["hi"], K)
            for name, cfg in PARAMS.items()
        }
        print_ranges(self.ranges)

    def normal_params(self) -> dict:
        return self._build(
            "normal",
            0,
            listprice_mode="normal",
            couponamt_mode="normal",
            wholesalecost_mode="normal",
        )

    def s1_listprice_params(self) -> dict:
        return self._build(
            "s1_listprice",
            1,
            listprice_mode="anomaly",
            couponamt_mode="normal",
            wholesalecost_mode="normal",
        )

    def s1_couponamt_params(self) -> dict:
        return self._build(
            "s1_couponamt",
            1,
            listprice_mode="normal",
            couponamt_mode="anomaly",
            wholesalecost_mode="normal",
        )

    def s1_wholesalecost_params(self) -> dict:
        return self._build(
            "s1_wholesalecost",
            1,
            listprice_mode="normal",
            couponamt_mode="normal",
            wholesalecost_mode="anomaly",
        )

    def s1_mix_params(self) -> dict:
        return self._build(
            "s1_mix",
            1,
            listprice_mode="anomaly",
            couponamt_mode="anomaly",
            wholesalecost_mode="anomaly",
        )

    def _build(
        self,
        scenario: str,
        label: int,
        listprice_mode: str,
        couponamt_mode: str,
        wholesalecost_mode: str,
    ) -> dict:
        lp = sample_six(self.ranges["listprice"], listprice_mode, self.rng)
        ca = sample_six(self.ranges["couponamt"], couponamt_mode, self.rng)
        wc = sample_six(self.ranges["wholesalecost"], wholesalecost_mode, self.rng)

        params = {
            "label": label,
            "scenario": scenario,
            "lp_window": LP_WINDOW,
            "ca_window": CA_WINDOW,
            "wc_window": WC_WINDOW,
        }

        for i in range(6):
            params[f"lp{i + 1}"] = lp[i]
            params[f"ca{i + 1}"] = ca[i]
            params[f"wc{i + 1}"] = wc[i]

        return params


# ============================================================
# RENDERING
# ============================================================

def render_sql(params: dict) -> str:
    return SQL_TEMPLATE.format(**params)


def render_validation(params: dict) -> str:
    return VALIDATION_TEMPLATE.format(**params)


def write_metadata(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ============================================================
# VALIDATION
# ============================================================

def validate_params(
    con: duckdb.DuckDBPyConnection,
    params: dict,
) -> tuple[bool, dict]:
    """
    Q28 always returns one row because each block is an aggregate without GROUP BY.
    Therefore, count(*) over the final query is not a useful validation.

    We validate that each block B1..B6 has at least one contributing tuple.
    """

    row = con.execute(render_validation(params)).fetchone()

    if row is None:
        return False, {}

    counts = [int(x) for x in row]

    valid = all(x > 0 for x in counts)

    count_meta = {
        "B1_CNT": counts[0],
        "B2_CNT": counts[1],
        "B3_CNT": counts[2],
        "B4_CNT": counts[3],
        "B5_CNT": counts[4],
        "B6_CNT": counts[5],
        "n_rows": 1,
    }

    return valid, count_meta


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
            valid, count_meta = validate_params(con, params)
        except Exception as e:
            print(f"[ERROR] validation failed for {params}: {e}")
            continue

        if not valid:
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
                **count_meta,
            }
        )

        i += 1

        if i % 1000 == 0:
            print(f"  [GEN] {prefix}: {i}/{n}")

    write_metadata(out_dir / "metadata.csv", rows)

    print(f"  [OK] {prefix}: {len(rows)} queries, attempts={attempts}")

    return rows


# ============================================================
# DIAGNOSTICS
# ============================================================

def values_from(df: pd.DataFrame, prefix: str) -> set[int]:
    cols = [f"{prefix}{i}" for i in range(1, 7)]
    out: set[int] = set()

    for c in cols:
        out |= set(df[c].astype(int).tolist())

    return out


def print_overlap_for_param(
    norm: pd.DataFrame,
    anom: pd.DataFrame,
    scenario: str,
    prefix: str,
    should_be_disjoint: bool,
) -> None:
    g = anom[anom["scenario"] == scenario]

    if g.empty:
        return

    norm_values = values_from(norm, prefix)
    anom_values = values_from(g, prefix)

    overlap = norm_values & anom_values

    ok = len(overlap) == 0 if should_be_disjoint else True

    print(f"  {prefix:<3} overlap={len(overlap):<4} {'OK' if ok else 'ERROR'}")

    if not ok:
        print(f"      overlap sample={sorted(overlap)[:10]}")


def print_count_stats(df: pd.DataFrame, label: str) -> None:
    print(f"\n[{label} BLOCK COUNTS]")
    for c in ["B1_CNT", "B2_CNT", "B3_CNT", "B4_CNT", "B5_CNT", "B6_CNT"]:
        print(
            f"  {c}: "
            f"min={int(df[c].min())} "
            f"median={int(df[c].median())} "
            f"max={int(df[c].max())}"
        )


def print_diagnostics(normal_rows: list[dict], anom_rows: list[dict]) -> None:
    norm = pd.DataFrame(normal_rows)
    anom = pd.DataFrame(anom_rows)

    print("\n" + "=" * 70)
    print("[DIAGNOSTICS]")
    print("=" * 70)

    print(f"\n[NORMAL] n={len(norm)}")
    for prefix, name in [
        ("lp", "listprice"),
        ("ca", "couponamt"),
        ("wc", "wholesalecost"),
    ]:
        vals = values_from(norm, prefix)
        print(f"  {name:<15} unique={len(vals):<5} min={min(vals)} max={max(vals)}")

    print_count_stats(norm, "NORMAL")

    print(f"\n[ANOMALY] n={len(anom)}")
    print(f"  scenarios={anom['scenario'].value_counts().to_dict()}")

    for scenario in sorted(anom["scenario"].unique()):
        g = anom[anom["scenario"] == scenario]

        print(f"\n[{scenario}] n={len(g)}")

        for prefix, name in [
            ("lp", "listprice"),
            ("ca", "couponamt"),
            ("wc", "wholesalecost"),
        ]:
            vals = values_from(g, prefix)
            print(
                f"  {name:<15} unique={len(vals):<5} "
                f"min={min(vals)} max={max(vals)}"
            )

        if scenario == "s1_listprice":
            print_overlap_for_param(norm, anom, scenario, "lp", True)
            print_overlap_for_param(norm, anom, scenario, "ca", False)
            print_overlap_for_param(norm, anom, scenario, "wc", False)

        elif scenario == "s1_couponamt":
            print_overlap_for_param(norm, anom, scenario, "lp", False)
            print_overlap_for_param(norm, anom, scenario, "ca", True)
            print_overlap_for_param(norm, anom, scenario, "wc", False)

        elif scenario == "s1_wholesalecost":
            print_overlap_for_param(norm, anom, scenario, "lp", False)
            print_overlap_for_param(norm, anom, scenario, "ca", False)
            print_overlap_for_param(norm, anom, scenario, "wc", True)

        elif scenario == "s1_mix":
            print_overlap_for_param(norm, anom, scenario, "lp", True)
            print_overlap_for_param(norm, anom, scenario, "ca", True)
            print_overlap_for_param(norm, anom, scenario, "wc", True)

        print_count_stats(g, scenario)

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q28 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q28.")
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
    for name in ['q28_normal', 'q28_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q28_normal"
    OUT_ANOM = args.output_dir / "q28_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print(f"[DB] {DB_PATH}  schema={DB_SCHEMA}")
    print(f"[CONFIG] Q28")
    print(f"[CONFIG] K={K}")
    print(f"[CONFIG] N_NORMAL={N_NORMAL}")
    print(f"[CONFIG] N_ANOMALY_PER_TYPE={N_ANOMALY_PER_TYPE}")
    print(f"[CONFIG] generate_mix={GENERATE_MIX}")
    print(f"[CONFIG] windows: LP={LP_WINDOW}, CA={CA_WINDOW}, WC={WC_WINDOW}")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET search_path = ?", [DB_SCHEMA])

    rng = np.random.default_rng(SEED)
    sampler = QuerySampler(rng)

    print("\n[GEN] normal")
    normal_rows = write_queries(
        OUT_NORMAL,
        "normal",
        N_NORMAL,
        sampler.normal_params,
        con,
    )

    anom_rows: list[dict] = []

    print("\n[GEN] s1_listprice")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_listprice",
            N_ANOMALY_PER_TYPE,
            sampler.s1_listprice_params,
            con,
        )
    )

    print("\n[GEN] s1_couponamt")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_couponamt",
            N_ANOMALY_PER_TYPE,
            sampler.s1_couponamt_params,
            con,
        )
    )

    print("\n[GEN] s1_wholesalecost")
    anom_rows.extend(
        write_queries(
            OUT_ANOM,
            "s1_wholesalecost",
            N_ANOMALY_PER_TYPE,
            sampler.s1_wholesalecost_params,
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

    write_metadata(OUT_ANOM / "metadata.csv", anom_rows)

    con.close()

    print_diagnostics(normal_rows, anom_rows)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
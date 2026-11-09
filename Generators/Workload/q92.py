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

# K=0.5 -> normal mais compacto
# K=1.0 -> normal central padrão
# K>1   -> normal mais largo
K = 1.0

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

OUT_NORMAL = None  # Set by command-line arguments.
OUT_ANOM = None  # Set by command-line arguments.


# ============================================================
# MANUFACTURER CONFIG
# ============================================================

MANUFACT_LO = 1
MANUFACT_HI = 1000


# ============================================================
# DATE CONFIG
# ============================================================
# Alinhado com o padrão temporal comum:
#   normal  = contexto recente
#   anomaly = contexto histórico disjunto
#
# A query usa uma janela de 90 dias.
# Logo o início máximo é context_max - 90 dias.

DATE_WINDOW_DAYS = 90

NORMAL_CONTEXT_MIN = date(2002, 1, 1)
NORMAL_CONTEXT_MAX = date(2002, 6, 30)

ANOM_CONTEXT_MIN = date(2001, 7, 1)
ANOM_CONTEXT_MAX = date(2001, 12, 31)


# ============================================================
# SQL
# ============================================================

SQL_TEMPLATE = """
WITH avg_discount AS (
    SELECT
        ws_avg.ws_item_sk AS item_sk,
        1.3 * avg(ws_avg.ws_ext_discount_amt) AS discount_threshold,
        array_agg(
            list_value(
                ws_avg.prov,
                d_avg.prov
            )
        ) AS avg_prov
    FROM web_sales ws_avg
    JOIN date_dim d_avg
      ON d_avg.d_date_sk = ws_avg.ws_sold_date_sk
    WHERE d_avg.d_date BETWEEN cast('{wsdate}' AS date)
                           AND cast('{wsdate}' AS date) + INTERVAL {date_window_days} DAY
    GROUP BY
        ws_avg.ws_item_sk
),

qualified AS (
    SELECT
        ws.ws_ext_discount_amt,
        list_value(
            ws.prov,
            item.prov,
            d.prov
        ) AS outer_prov,
        ad.avg_prov AS avg_prov
    FROM web_sales ws
    JOIN item
      ON item.i_item_sk = ws.ws_item_sk
    JOIN date_dim d
      ON d.d_date_sk = ws.ws_sold_date_sk
    JOIN avg_discount ad
      ON ad.item_sk = item.i_item_sk
    WHERE item.i_manufact_id = {manufact}
      AND d.d_date BETWEEN cast('{wsdate}' AS date)
                       AND cast('{wsdate}' AS date) + INTERVAL {date_window_days} DAY
      AND ws.ws_ext_discount_amt > ad.discount_threshold
)

SELECT
    sum(ws_ext_discount_amt) AS "Excess Discount Amount",

    -- Factorised provenance:
    -- each element is [outer witness factor, avg witness factor]
    array_agg(
        list_value(
            list_value(outer_prov),
            avg_prov
        )
    ) AS prov

FROM qualified
ORDER BY
    sum(ws_ext_discount_amt)
LIMIT 100
"""


COUNT_TEMPLATE = """
WITH avg_discount AS (
    SELECT
        ws_avg.ws_item_sk AS item_sk,
        1.3 * avg(ws_avg.ws_ext_discount_amt) AS discount_threshold
    FROM web_sales ws_avg
    JOIN date_dim d_avg
      ON d_avg.d_date_sk = ws_avg.ws_sold_date_sk
    WHERE d_avg.d_date BETWEEN cast('{wsdate}' AS date)
                           AND cast('{wsdate}' AS date) + INTERVAL {date_window_days} DAY
    GROUP BY
        ws_avg.ws_item_sk
)

SELECT count(*) AS n_rows
FROM web_sales ws
JOIN item
  ON item.i_item_sk = ws.ws_item_sk
JOIN date_dim d
  ON d.d_date_sk = ws.ws_sold_date_sk
JOIN avg_discount ad
  ON ad.item_sk = item.i_item_sk
WHERE item.i_manufact_id = {manufact}
  AND d.d_date BETWEEN cast('{wsdate}' AS date)
                   AND cast('{wsdate}' AS date) + INTERVAL {date_window_days} DAY
  AND ws.ws_ext_discount_amt > ad.discount_threshold
"""


# ============================================================
# DATE HELPERS
# ============================================================

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
# CONTINUOUS MANUFACTURER RANGE
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


def sample_normal_manufact(
    r: dict,
    rng: np.random.Generator,
) -> int:
    sigma_eff = max(
        (r["hi_normal"] - r["lo_normal"]) / 6.0,
        1e-6,
    )

    return sample_truncated_normal(
        r["mu"],
        sigma_eff,
        r["lo_normal"],
        r["hi_normal"],
        rng,
    )


def sample_anomaly_manufact(
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


def print_manufact_range(r: dict) -> None:
    print("\n[MANUFACTURER]")
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
# SAMPLER
# ============================================================

class QuerySampler:
    def __init__(
        self,
        rng: np.random.Generator,
    ):
        self.rng = rng

        self.manufact_range = compute_continuous_range(
            MANUFACT_LO,
            MANUFACT_HI,
            K,
        )

        print_manufact_range(self.manufact_range)

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
        print(f"  normal wsdate=[{self.normal_date_lo}, {self.normal_date_hi}]")
        print(f"  anomaly context=[{ANOM_CONTEXT_MIN}, {ANOM_CONTEXT_MAX}]")
        print(f"  anomaly wsdate=[{self.anom_date_lo}, {self.anom_date_hi}]")

    def normal_params(self) -> dict:
        return self._build(
            scenario="normal",
            label=0,
            manufact_mode="normal",
            date_mode="normal",
        )

    def s1_manufact_params(self) -> dict:
        return self._build(
            scenario="s1_manufact",
            label=1,
            manufact_mode="anomaly",
            date_mode="normal",
        )

    def s1_date_params(self) -> dict:
        return self._build(
            scenario="s1_date",
            label=1,
            manufact_mode="normal",
            date_mode="anomaly",
        )

    def s1_mix_params(self) -> dict:
        return self._build(
            scenario="s1_mix",
            label=1,
            manufact_mode="anomaly",
            date_mode="anomaly",
        )

    def _sample_manufact(self, mode: str) -> int:
        if mode == "normal":
            return sample_normal_manufact(
                self.manufact_range,
                self.rng,
            )

        if mode == "anomaly":
            return sample_anomaly_manufact(
                self.manufact_range,
                self.rng,
            )

        raise ValueError(f"Modo de manufacturer inválido: {mode}")

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

    def _build(
        self,
        scenario: str,
        label: int,
        manufact_mode: str,
        date_mode: str,
    ) -> dict:
        manufact = self._sample_manufact(manufact_mode)
        wsdate = self._sample_date(date_mode)

        return {
            "label": label,
            "scenario": scenario,
            "manufact": int(manufact),
            "wsdate": wsdate.isoformat(),
            "date_window_days": DATE_WINDOW_DAYS,
            "provenance_semantics": "factorized_avg_predicate",
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

    if param == "manufact":
        norm_values = set(norm["manufact"].astype(int))
        anom_values = set(g["manufact"].astype(int))
    elif param == "date":
        norm_values = set(norm["wsdate"].astype(str))
        anom_values = set(g["wsdate"].astype(str))
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
    print(
        f"  manufact unique={norm['manufact'].nunique()} "
        f"min={int(norm['manufact'].min())} "
        f"max={int(norm['manufact'].max())} "
        f"mean={norm['manufact'].mean():.2f} "
        f"std={norm['manufact'].std():.2f}"
    )
    print(
        f"  wsdate unique={norm['wsdate'].nunique()} "
        f"min={norm['wsdate'].min()} "
        f"max={norm['wsdate'].max()}"
    )
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
        print(
            f"  manufact unique={g['manufact'].nunique()} "
            f"min={int(g['manufact'].min())} "
            f"max={int(g['manufact'].max())}"
        )
        print(
            f"  wsdate unique={g['wsdate'].nunique()} "
            f"min={g['wsdate'].min()} "
            f"max={g['wsdate'].max()}"
        )
        print(
            f"  n_rows min={g['n_rows'].min()} "
            f"median={int(g['n_rows'].median())} "
            f"max={g['n_rows'].max()}"
        )

        if scenario == "s1_manufact":
            print_overlap_for_param(norm, anom, scenario, "manufact", True)
            print_overlap_for_param(norm, anom, scenario, "date", False)

        elif scenario == "s1_date":
            print_overlap_for_param(norm, anom, scenario, "manufact", False)
            print_overlap_for_param(norm, anom, scenario, "date", True)

        elif scenario == "s1_mix":
            print_overlap_for_param(norm, anom, scenario, "manufact", True)
            print_overlap_for_param(norm, anom, scenario, "date", True)

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q92 normal and S1 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q92.")
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
    for name in ['q92_normal', 'q92_anomalies']:
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
    OUT_NORMAL = args.output_dir / "q92_normal"
    OUT_ANOM = args.output_dir / "q92_anomalies"
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
    print("  - manufacturer id")
    print("  - web sales date")
    print("[CONFIG] final provenance semantics:")
    print("  - factorized AVG predicate provenance")
    print(f"[CONFIG] generate_mix={GENERATE_MIX}")
    print(f"[CONFIG] output base={EXTERNAL_BASE_DIR}")

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

    print_diagnostics(
        normal_rows,
        anom_rows,
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
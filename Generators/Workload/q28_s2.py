#!/usr/bin/env python3
from __future__ import annotations

import argparse

import csv
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DB_PATH = None  # Set by command-line arguments.
DB_SCHEMA = "tpcds"

SEED = 42

# Gerar mais do que o necessário para permitir sampling posterior.
# 500 por subtipo -> 2000 S2 estruturais para Q28.
N_S2_PER_TYPE = 500

# Metadata dos normais já gerados.
# Usada para amostrar parâmetros da distribuição normal real.
Q28_NORMAL_DIR = None  # Set by command-line arguments.

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.
OUT_S2 = None  # Set by command-line arguments.

# Fallback caso não exista metadata.csv dos normais.
K = 1.0

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

NORMAL_Q_RANGES = [
    (0, 5),
    (6, 10),
    (11, 15),
    (16, 20),
    (21, 25),
    (26, 30),
]

S2_SUBTYPES = [
    "q28_s2_join_add_item",
    "q28_s2_drop_listprice_predicate",
    "q28_s2_drop_wholesalecost_predicate",
    "q28_s2_block_remove_b6",
]

S2_INFO: Dict[str, dict] = {
    "q28_s2_join_add_item": {
        "label": 2,
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "intent_preservation": "high",
        "description": (
            "Adds item table inside each quantity block using "
            "store_sales.ss_item_sk = item.i_item_sk; each block witness "
            "therefore includes item provenance in addition to store_sales."
        ),
    },
    "q28_s2_drop_listprice_predicate": {
        "label": 2,
        "category": "s2_predicate_component",
        "logical_unit": "predicate component removal",
        "intent_preservation": "medium",
        "description": (
            "Removes the ss_list_price component from the disjunctive "
            "predicate inside each block, preserving coupon and wholesale-cost "
            "components with normal parameter values."
        ),
    },
    "q28_s2_drop_wholesalecost_predicate": {
        "label": 2,
        "category": "s2_predicate_component",
        "logical_unit": "predicate component removal",
        "intent_preservation": "medium",
        "description": (
            "Removes the ss_wholesale_cost component from the disjunctive "
            "predicate inside each block, preserving list-price and coupon "
            "components with normal parameter values."
        ),
    },
    "q28_s2_block_remove_b6": {
        "label": 2,
        "category": "s2_block_structure",
        "logical_unit": "multi-block query structure",
        "intent_preservation": "medium_low",
        "description": (
            "Removes the sixth quantity block B6 from the final query. "
            "This is a stronger structural mutation because it changes the "
            "six-block composition of Q28."
        ),
    },
}


# ============================================================
# PARAMETER SAMPLING
# ============================================================

def compute_range(lo: int, hi: int, k: float) -> dict:
    mu = (lo + hi) / 2.0
    sigma = (hi - lo) / 6.0

    lo_normal = max(lo, int(np.floor(mu - k * sigma)))
    hi_normal = min(hi, int(np.ceil(mu + k * sigma)))

    return {
        "lo": lo,
        "hi": hi,
        "mu": mu,
        "sigma": sigma,
        "lo_normal": lo_normal,
        "hi_normal": hi_normal,
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


def sample_normal_value(r: dict, rng: np.random.Generator) -> int:
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


def sample_six(r: dict, rng: np.random.Generator) -> list[int]:
    return [
        sample_normal_value(r, rng)
        for _ in range(6)
    ]


def fallback_sample_params(rng: np.random.Generator) -> dict:
    ranges = {
        name: compute_range(cfg["lo"], cfg["hi"], K)
        for name, cfg in PARAMS.items()
    }

    lp = sample_six(ranges["listprice"], rng)
    ca = sample_six(ranges["couponamt"], rng)
    wc = sample_six(ranges["wholesalecost"], rng)

    params = {
        "param_source": "fallback_gaussian_normal",
        "lp_window": LP_WINDOW,
        "ca_window": CA_WINDOW,
        "wc_window": WC_WINDOW,
    }

    for i in range(6):
        params[f"lp{i + 1}"] = int(lp[i])
        params[f"ca{i + 1}"] = int(ca[i])
        params[f"wc{i + 1}"] = int(wc[i])

    return params


def load_normal_metadata(path: Path) -> Optional[pd.DataFrame]:
    meta_path = path / "metadata.csv"

    if not meta_path.exists():
        return None

    df = pd.read_csv(meta_path)

    required = []

    for i in range(1, 7):
        required.extend(
            [
                f"lp{i}",
                f"ca{i}",
                f"wc{i}",
            ]
        )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        print(f"[WARN] metadata.csv missing columns: {missing}")
        return None

    df = df.dropna(subset=required).copy()

    for c in required:
        df[c] = df[c].astype(int)

    if "lp_window" not in df.columns:
        df["lp_window"] = LP_WINDOW

    if "ca_window" not in df.columns:
        df["ca_window"] = CA_WINDOW

    if "wc_window" not in df.columns:
        df["wc_window"] = WC_WINDOW

    df["lp_window"] = df["lp_window"].astype(int)
    df["ca_window"] = df["ca_window"].astype(int)
    df["wc_window"] = df["wc_window"].astype(int)

    if df.empty:
        return None

    return df


class Q28S2Sampler:
    def __init__(
        self,
        normal_metadata_dir: Path,
        seed: int,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.normal_meta = load_normal_metadata(normal_metadata_dir)

        if self.normal_meta is not None:
            print(f"[PARAMS] using normal metadata: {normal_metadata_dir / 'metadata.csv'}")
            print(f"[PARAMS] normal metadata rows={len(self.normal_meta):,}")
        else:
            print("[PARAMS] normal metadata unavailable; using fallback Gaussian sampler")

    def sample_params(self) -> dict:
        if self.normal_meta is None:
            return fallback_sample_params(self.rng)

        idx = int(
            self.rng.choice(
                self.normal_meta.index.to_numpy(),
                size=1,
                replace=True,
            )[0]
        )

        row = self.normal_meta.loc[idx]

        params = {
            "param_source": "normal_metadata",
            "lp_window": int(row["lp_window"]),
            "ca_window": int(row["ca_window"]),
            "wc_window": int(row["wc_window"]),
        }

        for i in range(1, 7):
            params[f"lp{i}"] = int(row[f"lp{i}"])
            params[f"ca{i}"] = int(row[f"ca{i}"])
            params[f"wc{i}"] = int(row[f"wc{i}"])

        return params

    def sample(self, subtype: str) -> dict:
        info = S2_INFO[subtype]
        params = self.sample_params()

        params.update(
            {
                "label": int(info["label"]),
                "scenario": subtype,
                "category": info["category"],
                "logical_unit": info["logical_unit"],
                "intent_preservation": info["intent_preservation"],
                "description": info["description"],
            }
        )

        return params


# ============================================================
# SQL BUILDER
# ============================================================

def predicate_for_block(i: int, subtype: str) -> str:
    lp = f"ss_list_price between {{lp{i}}} and {{lp{i}}}+{{lp_window}}"
    ca = f"ss_coupon_amt between {{ca{i}}} and {{ca{i}}}+{{ca_window}}"
    wc = f"ss_wholesale_cost between {{wc{i}}} and {{wc{i}}}+{{wc_window}}"

    if subtype == "q28_s2_drop_listprice_predicate":
        return f"({ca} or {wc})"

    if subtype == "q28_s2_drop_wholesalecost_predicate":
        return f"({lp} or {ca})"

    return f"({lp} or {ca} or {wc})"


def block_ranges_for_subtype(subtype: str) -> list[tuple[int, int]]:
    if subtype == "q28_s2_block_remove_b6":
        return NORMAL_Q_RANGES[:5]

    return NORMAL_Q_RANGES


def make_from_where(
    i: int,
    qlo: int,
    qhi: int,
    subtype: str,
) -> tuple[str, str]:
    pred = predicate_for_block(i, subtype)

    if subtype == "q28_s2_join_add_item":
        from_clause = "store_sales, item"
        where_clause = f"""
      where store_sales.ss_item_sk = item.i_item_sk
        and ss_quantity between {qlo} and {qhi}
        and {pred}
"""
        return from_clause, where_clause

    from_clause = "store_sales"
    where_clause = f"""
      where ss_quantity between {qlo} and {qhi}
        and {pred}
"""
    return from_clause, where_clause


def prov_expr_for_subtype(subtype: str) -> str:
    """
    Q28 processing treats each block provenance as one aggregate witness.
    For the join-add mutation, we concatenate store_sales.prov and item.prov
    for every contributing joined tuple, then flatten the block into one list.
    This keeps the same final Q28 shape: one witness-like object per block.
    """

    if subtype == "q28_s2_join_add_item":
        return """
            list_reduce(
                array_agg(list_value(store_sales.prov, item.prov)),
                (x, y) -> list_concat(x, y)
            ) prov
"""

    return "array_agg(store_sales.prov) prov"


def build_sql(subtype: str) -> str:
    blocks = []
    aliases = []

    ranges = block_ranges_for_subtype(subtype)

    for idx, (qlo, qhi) in enumerate(ranges, start=1):
        from_clause, where_clause = make_from_where(
            i=idx,
            qlo=qlo,
            qhi=qhi,
            subtype=subtype,
        )

        prov_expr = prov_expr_for_subtype(subtype)

        block = f"""
     (select avg(ss_list_price) B{idx}_LP
            ,count(ss_list_price) B{idx}_CNT
            ,count(distinct ss_list_price) B{idx}_CNTD
            ,{prov_expr}
      from {from_clause}
{where_clause}) B{idx}
"""
        blocks.append(block)
        aliases.append(f"B{idx}.prov")

    return f"""
select *,
       list_value({", ".join(aliases)}) as prov
from
{",".join(blocks)}
limit 100
"""


def build_validation_sql(subtype: str) -> str:
    blocks = []
    aliases = []

    ranges = block_ranges_for_subtype(subtype)

    for idx, (qlo, qhi) in enumerate(ranges, start=1):
        from_clause, where_clause = make_from_where(
            i=idx,
            qlo=qlo,
            qhi=qhi,
            subtype=subtype,
        )

        block = f"""
     (select count(ss_list_price) B{idx}_CNT
      from {from_clause}
{where_clause}) B{idx}
"""
        blocks.append(block)
        aliases.append(f"B{idx}_CNT")

    return f"""
select
    {", ".join(aliases)}
from
{",".join(blocks)}
"""


def render_sql(
    template: str,
    params: dict,
) -> str:
    return template.format(**params)


# ============================================================
# VALIDATION
# ============================================================

def validate_params(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    params: dict,
) -> tuple[bool, dict]:
    """
    Q28 always returns one row because each block is an aggregate without GROUP BY.
    Therefore final count(*) is not informative.

    We validate using each block count. A generated S2 instance is kept only
    if every retained block has at least one contributing tuple.
    """

    validation_sql = build_validation_sql(subtype)
    row = con.execute(render_sql(validation_sql, params)).fetchone()

    if row is None:
        return False, {}

    counts = [int(x) for x in row]

    valid = all(x > 0 for x in counts)

    meta = {
        "n_blocks": len(counts),
        "n_rows": 1,
        "block_min": int(min(counts)) if counts else 0,
        "block_median": float(np.median(counts)) if counts else 0.0,
        "block_max": int(max(counts)) if counts else 0,
        "block_sum": int(sum(counts)),
        "zero_blocks": int(sum(1 for x in counts if x == 0)),
    }

    for i, c in enumerate(counts, start=1):
        meta[f"B{i}_CNT"] = int(c)

    # Para block_remove_b6, B6 não existe. Guardamos vazio para manter schema
    # parecido com os restantes.
    for i in range(len(counts) + 1, 7):
        meta[f"B{i}_CNT"] = ""

    return valid, meta


def is_duplicate_sql(
    sql: str,
    seen: set[str],
) -> bool:
    key = " ".join(sql.split())

    if key in seen:
        return True

    seen.add(key)
    return False


# ============================================================
# WRITER
# ============================================================

def write_metadata(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    fields = sorted(
        {
            k
            for row in rows
            for k in row.keys()
        }
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_queries_for_subtype(
    subtype: str,
    n: int,
    sampler: Q28S2Sampler,
    con: duckdb.DuckDBPyConnection,
    global_rows: list[dict],
) -> None:
    OUT_S2.mkdir(parents=True, exist_ok=True)

    sql_template = build_sql(subtype)

    short = subtype.replace("q28_s2_", "")
    short = short.replace("_", "")

    seen_sql: set[str] = set()

    i = 0
    attempts = 0

    max_attempts = max(100_000, n * 500)

    print(f"\n[GEN] {subtype}")
    print(f"      target={n}")

    while i < n:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"Too many attempts for {subtype}: "
                f"generated {i}/{n} after {attempts} attempts."
            )

        params = sampler.sample(subtype)

        try:
            valid, count_meta = validate_params(
                con=con,
                subtype=subtype,
                params=params,
            )
        except Exception as e:
            print(f"[WARN] validation failed for {subtype}: {e}")
            continue

        if not valid:
            continue

        sql = render_sql(sql_template, params).strip() + "\n"

        if is_duplicate_sql(sql, seen_sql):
            continue

        filename = f"{short}_{i:05d}.sql"

        (OUT_S2 / filename).write_text(
            sql,
            encoding="utf-8",
        )

        row = {
            "file": filename,
            **params,
            **count_meta,
        }

        global_rows.append(row)

        i += 1

        if i % 100 == 0:
            print(f"  [GEN] {subtype}: {i}/{n}")

    print(
        f"  [OK] {subtype}: {i} queries, attempts={attempts}"
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def values_from(
    df: pd.DataFrame,
    prefix: str,
) -> set[int]:
    cols = [
        f"{prefix}{i}"
        for i in range(1, 7)
        if f"{prefix}{i}" in df.columns
    ]

    out: set[int] = set()

    for c in cols:
        vals = pd.to_numeric(
            df[c],
            errors="coerce",
        ).dropna().astype(int).tolist()

        out |= set(vals)

    return out


def print_param_summary(
    df: pd.DataFrame,
    label: str,
) -> None:
    print(f"\n[{label} PARAMS] n={len(df):,}")

    for prefix, name in [
        ("lp", "listprice"),
        ("ca", "couponamt"),
        ("wc", "wholesalecost"),
    ]:
        vals = values_from(df, prefix)

        if not vals:
            print(f"  {name:<15} no values")
            continue

        print(
            f"  {name:<15} unique={len(vals):<5} "
            f"min={min(vals)} max={max(vals)}"
        )


def print_count_summary(
    df: pd.DataFrame,
    label: str,
) -> None:
    print(f"\n[{label} BLOCK COUNTS]")

    for c in [
        "block_min",
        "block_median",
        "block_max",
        "block_sum",
        "zero_blocks",
        "n_blocks",
    ]:
        if c not in df.columns:
            continue

        s = pd.to_numeric(
            df[c],
            errors="coerce",
        ).dropna()

        if s.empty:
            continue

        print(
            f"  {c:<14} "
            f"min={s.min():.0f} "
            f"median={s.median():.0f} "
            f"max={s.max():.0f}"
        )


def print_diagnostics(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("[DIAGNOSTICS]")
    print("=" * 80)

    print(f"\n[TOTAL] {len(df):,} S2 queries")
    print(f"[SCENARIOS] {df['scenario'].value_counts().to_dict()}")

    print_param_summary(df, "ALL S2")

    for subtype in S2_SUBTYPES:
        g = df[df["scenario"] == subtype]

        if g.empty:
            continue

        print("\n" + "-" * 80)
        print(f"[{subtype}] n={len(g):,}")
        print(f"  category={S2_INFO[subtype]['category']}")
        print(f"  logical_unit={S2_INFO[subtype]['logical_unit']}")
        print(f"  intent_preservation={S2_INFO[subtype]['intent_preservation']}")

        print_param_summary(g, subtype)
        print_count_summary(g, subtype)

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q28 S2 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q28.")
    p.add_argument("--schema", default=DB_SCHEMA)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-per-type", type=int, default=N_S2_PER_TYPE)
    p.add_argument("--normal-dir", type=Path, required=True, help="Original normal pool with its SQL and metadata, as required by this generator.")
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
    args.normal_dir = args.normal_dir.expanduser().resolve()
    if not args.normal_dir.is_dir():
        p.error(f"Normal pool does not exist: {args.normal_dir}")
    for name in ['q28_s2_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, EXTERNAL_BASE_DIR, N_S2_PER_TYPE, OUT_S2, Q28_NORMAL_DIR, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_S2_PER_TYPE = args.n_per_type
    Q28_NORMAL_DIR = args.normal_dir
    OUT_S2 = args.output_dir / "q28_s2_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir


def main() -> None:
    configure_cli()
    print("=" * 90)
    print("[Q28 S2 GENERATOR]")
    print("=" * 90)
    print(f"DB_PATH={DB_PATH}")
    print(f"DB_SCHEMA={DB_SCHEMA}")
    print(f"Q28_NORMAL_DIR={Q28_NORMAL_DIR}")
    print(f"OUT_S2={OUT_S2}")
    print(f"N_S2_PER_TYPE={N_S2_PER_TYPE}")
    print(f"S2_SUBTYPES={S2_SUBTYPES}")
    print("=" * 90)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("SET search_path = ?", [DB_SCHEMA])

    sampler = Q28S2Sampler(
        normal_metadata_dir=Q28_NORMAL_DIR,
        seed=SEED,
    )

    OUT_S2.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for subtype in S2_SUBTYPES:
        write_queries_for_subtype(
            subtype=subtype,
            n=N_S2_PER_TYPE,
            sampler=sampler,
            con=con,
            global_rows=all_rows,
        )

    write_metadata(
        OUT_S2 / "metadata.csv",
        all_rows,
    )

    con.close()

    print_diagnostics(all_rows)

    print("\n[DONE]")
    print(f"  queries   : {len(all_rows):,}")
    print(f"  directory : {OUT_S2}")
    print(f"  metadata  : {OUT_S2 / 'metadata.csv'}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

"""
Main S2 subtypes
----------------
- q84_s2_join_add_item
- q84_s2_join_add_reason
- q84_s2_where_remove_income_lower
- q84_s2_where_remove_income_upper
- q84_s2_limit_1
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DB_PATH = None  # Set by command-line arguments.
DB_SCHEMA = "tpcds"

SEED = 42
N_S2_PER_TYPE = 500

EXTERNAL_BASE_DIR = None  # Set by command-line arguments.

Q84_NORMAL_DIR = None  # Set by command-line arguments.
OUT_S2 = None  # Set by command-line arguments.

N_PROV_SHAPE_CHECKS_PER_SUBTYPE = 3
MAX_WITNESSES_CHECKED_PER_QUERY = 100

DEFAULT_INCOME_WINDOW = 50_000

S2_SUBTYPES = [
    "q84_s2_join_add_item",
    "q84_s2_join_add_reason",
    "q84_s2_where_remove_income_lower",
    "q84_s2_where_remove_income_upper",
    "q84_s2_limit_1",
]

S2_INFO: Dict[str, dict] = {
    "q84_s2_join_add_item": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_witness_size": 7,
        "description": (
            "Adds item through store_returns.sr_item_sk = item.i_item_sk "
            "and appends item provenance to each witness."
        ),
    },
    "q84_s2_join_add_reason": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_join",
        "logical_unit": "join/table addition",
        "expected_witness_size": 7,
        "description": (
            "Adds reason through store_returns.sr_reason_sk = "
            "reason.r_reason_sk and appends reason provenance to each witness."
        ),
    },
    "q84_s2_where_remove_income_lower": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "expected_witness_size": 6,
        "description": (
            "Removes ib_lower_bound >= income while preserving "
            "ib_upper_bound <= income + income_window."
        ),
    },
    "q84_s2_where_remove_income_upper": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_predicate",
        "logical_unit": "selection predicate removal",
        "expected_witness_size": 6,
        "description": (
            "Removes ib_upper_bound <= income + income_window while preserving "
            "ib_lower_bound >= income."
        ),
    },
    "q84_s2_limit_1": {
        "label": 1,
        "anomaly_class": "S2",
        "category": "s2_limit",
        "logical_unit": "result cardinality limit",
        "expected_witness_size": 6,
        "description": (
            "Replaces LIMIT 100 with LIMIT 1 without filtering normal source "
            "instances according to their result cardinality."
        ),
    },
}


# ============================================================
# NORMAL METADATA
# ============================================================


def load_normal_metadata() -> pd.DataFrame:
    metadata_path = Q84_NORMAL_DIR / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Normal Q84 metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required = ["file", "city", "income"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Normal metadata is missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    if df.empty:
        raise ValueError("Normal Q84 metadata contains no usable rows.")

    df["file"] = df["file"].astype(str)
    df["city"] = df["city"].astype(str)
    df["income"] = pd.to_numeric(
        df["income"],
        errors="raise",
    ).astype(int)

    if "income_window" not in df.columns:
        df["income_window"] = DEFAULT_INCOME_WINDOW

    df["income_window"] = pd.to_numeric(
        df["income_window"],
        errors="raise",
    ).astype(int)

    # A normal execution can enter the source population only once.
    df = df.drop_duplicates(subset=["file"]).reset_index(drop=True)

    return df


def select_common_normal_sources() -> pd.DataFrame:
    """
    Select one deterministic, unconditioned sample shared by all subtypes.

    No output statistic is used in this selection.
    """

    df = load_normal_metadata()

    if len(df) < N_S2_PER_TYPE:
        raise ValueError(
            f"Need {N_S2_PER_TYPE} distinct normal executions, "
            f"but metadata contains only {len(df)}."
        )

    rng = np.random.default_rng(SEED)

    positions = rng.choice(
        len(df),
        size=N_S2_PER_TYPE,
        replace=False,
    )

    selected = df.iloc[positions].copy().reset_index(drop=True)
    selected["_selection_index"] = np.arange(
        len(selected),
        dtype=np.int64,
    )

    print(
        f"[SOURCES] selected={len(selected):,} from available={len(df):,} seed={SEED}"
    )

    return selected


# ============================================================
# SQL
# ============================================================


def sql_escape(value: str) -> str:
    return str(value).replace("'", "''")


def prepare_params(row: pd.Series | dict) -> dict:
    return {
        "city": sql_escape(str(row["city"])),
        "income": int(row["income"]),
        "income_window": int(row["income_window"]),
    }


def query_spec(subtype: str) -> dict:
    relations = [
        "customer c",
        "customer_address ca",
        "customer_demographics cd",
        "household_demographics hd",
        "income_band ib",
        "store_returns sr",
    ]

    provenance = [
        "c.prov",
        "ca.prov",
        "cd.prov",
        "hd.prov",
        "ib.prov",
        "sr.prov",
    ]

    predicates = [
        "ca.ca_city = '{city}'",
        "c.c_current_addr_sk = ca.ca_address_sk",
    ]

    if subtype != "q84_s2_where_remove_income_lower":
        predicates.append("ib.ib_lower_bound >= {income}")

    if subtype != "q84_s2_where_remove_income_upper":
        predicates.append("ib.ib_upper_bound <= {income} + {income_window}")

    predicates.extend(
        [
            "ib.ib_income_band_sk = hd.hd_income_band_sk",
            "cd.cd_demo_sk = c.c_current_cdemo_sk",
            "hd.hd_demo_sk = c.c_current_hdemo_sk",
            "sr.sr_cdemo_sk = cd.cd_demo_sk",
        ]
    )

    if subtype == "q84_s2_join_add_item":
        relations.append("item i")
        provenance.append("i.prov")
        predicates.append("sr.sr_item_sk = i.i_item_sk")

    elif subtype == "q84_s2_join_add_reason":
        relations.append("reason r")
        provenance.append("r.prov")
        predicates.append("sr.sr_reason_sk = r.r_reason_sk")

    if subtype == "q84_s2_limit_1":
        limit = 1
    else:
        limit = 100

    return {
        "relations": relations,
        "provenance": provenance,
        "predicates": predicates,
        "limit": limit,
    }


def build_sql(
    subtype: str,
    with_provenance: bool,
    apply_limit: bool,
) -> str:
    spec = query_spec(subtype)

    relations_sql = ",\n     ".join(spec["relations"])
    predicates_sql = "\n  and ".join(spec["predicates"])

    if with_provenance:
        provenance_sql = ",\n        ".join(spec["provenance"])

        prov_select = f""",
    list_value(
        list_value(
            {provenance_sql}
        )
    ) as prov"""
    else:
        prov_select = ""

    limit_sql = ""

    if apply_limit:
        limit_sql = f"\nlimit {int(spec['limit'])}"

    return f"""
select
    c.c_customer_id as customer_id,
    coalesce(c.c_last_name, '') || ', ' ||
        coalesce(c.c_first_name, '') as customername
    {prov_select}
from {relations_sql}
where {predicates_sql}
order by
    c.c_customer_id
{limit_sql}
"""


def render_sql(
    subtype: str,
    row: pd.Series | dict,
    with_provenance: bool = True,
    apply_limit: bool = True,
) -> str:
    template = build_sql(
        subtype=subtype,
        with_provenance=with_provenance,
        apply_limit=apply_limit,
    )

    return template.format(**prepare_params(row))


# ============================================================
# STATIC VALIDATION
# ============================================================


def normalised_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").lower().split())


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise ValueError(message)


def validate_structure(
    subtype: str,
    sql: str,
) -> None:
    lower = normalised_sql(sql)

    # S1 city remains present in every structural mutation.
    require(
        "ca.ca_city =" in lower,
        f"{subtype}: city predicate is missing.",
    )

    # The normally sampled income value remains present in every mutation.
    require(
        "{income}" not in lower,
        f"{subtype}: unresolved income placeholder.",
    )

    require(
        "sr.sr_cdemo_sk = cd.cd_demo_sk" in lower,
        f"{subtype}: central store_returns join is missing.",
    )

    require(
        "ib.ib_income_band_sk = hd.hd_income_band_sk" in lower,
        f"{subtype}: income-band join is missing.",
    )

    if subtype == "q84_s2_join_add_item":
        require(
            "item i" in lower,
            "join_add_item: item relation is missing.",
        )
        require(
            "sr.sr_item_sk = i.i_item_sk" in lower,
            "join_add_item: item join predicate is missing.",
        )
        require(
            "i.prov" in lower,
            "join_add_item: item provenance is missing.",
        )

    elif subtype == "q84_s2_join_add_reason":
        require(
            "reason r" in lower,
            "join_add_reason: reason relation is missing.",
        )
        require(
            "sr.sr_reason_sk = r.r_reason_sk" in lower,
            "join_add_reason: reason join predicate is missing.",
        )
        require(
            "r.prov" in lower,
            "join_add_reason: reason provenance is missing.",
        )

    elif subtype == "q84_s2_where_remove_income_lower":
        require(
            "ib.ib_lower_bound >=" not in lower,
            "where_remove_income_lower: lower predicate was not removed.",
        )
        require(
            "ib.ib_upper_bound <=" in lower,
            "where_remove_income_lower: upper predicate is missing.",
        )

    elif subtype == "q84_s2_where_remove_income_upper":
        require(
            "ib.ib_upper_bound <=" not in lower,
            "where_remove_income_upper: upper predicate was not removed.",
        )
        require(
            "ib.ib_lower_bound >=" in lower,
            "where_remove_income_upper: lower predicate is missing.",
        )

    elif subtype == "q84_s2_limit_1":
        require(
            lower.endswith("limit 1"),
            "limit_1: query does not end with LIMIT 1.",
        )

    else:
        raise ValueError(f"Unknown subtype: {subtype}")


# ============================================================
# EXECUTION VALIDATION
# ============================================================


def count_rows(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    row: pd.Series,
    apply_limit: bool,
) -> int:
    query = (
        render_sql(
            subtype=subtype,
            row=row,
            with_provenance=False,
            apply_limit=apply_limit,
        )
        .strip()
        .rstrip(";")
    )

    count_sql = f"""
select count(*)
from (
{query}
) q84_s2_count
"""

    result = con.execute(count_sql).fetchone()

    if result is None:
        raise RuntimeError(f"{subtype}: count query returned no row.")

    return int(result[0])


def flatten_tokens(value: Any) -> list[Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        output: list[Any] = []

        for element in value:
            output.extend(flatten_tokens(element))

        return output

    return [value]


def provenance_witness_sizes(
    df: pd.DataFrame,
) -> list[int]:
    if "prov" not in df.columns:
        raise RuntimeError("Result has no prov column.")

    sizes: list[int] = []

    for value in df["prov"].tolist():
        sizes.append(len(flatten_tokens(value)))

        if len(sizes) >= MAX_WITNESSES_CHECKED_PER_QUERY:
            break

    return sizes


def validate_full_provenance(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    row: pd.Series,
) -> None:
    sql = render_sql(
        subtype=subtype,
        row=row,
        with_provenance=True,
        apply_limit=True,
    )

    df = con.execute(sql).fetchdf()

    if df.empty:
        raise RuntimeError(f"{subtype}: full provenance query returned no rows.")

    sizes = provenance_witness_sizes(df)

    if not sizes:
        raise RuntimeError(f"{subtype}: no provenance witnesses were extracted.")

    expected = int(S2_INFO[subtype]["expected_witness_size"])
    observed = sorted(set(sizes))

    if observed != [expected]:
        raise RuntimeError(
            f"{subtype}: expected witness size {expected}, observed {observed}."
        )

    print(
        f"[PROV OK] {subtype} "
        f"source={row['file']} "
        f"checked={len(sizes)} "
        f"witness_size={expected}"
    )


def run_pre_generation_checks(
    con: duckdb.DuckDBPyConnection,
    sources: pd.DataFrame,
) -> None:
    n_checks = min(
        N_PROV_SHAPE_CHECKS_PER_SUBTYPE,
        len(sources),
    )

    for subtype in S2_SUBTYPES:
        print(f"\n[PRE-CHECK] {subtype}")

        for index in range(n_checks):
            row = sources.iloc[index]

            sql = render_sql(
                subtype=subtype,
                row=row,
                with_provenance=True,
                apply_limit=True,
            )

            validate_structure(subtype, sql)

            limited_nrows = count_rows(
                con,
                subtype,
                row,
                apply_limit=True,
            )

            if limited_nrows <= 0:
                raise RuntimeError(
                    f"{subtype}: source {row['file']} produced "
                    "an empty result during pre-check."
                )

            validate_full_provenance(
                con,
                subtype,
                row,
            )


# ============================================================
# METADATA
# ============================================================


def metadata_value(value: Any) -> Any:
    if pd.isna(value):
        return ""

    if isinstance(value, np.generic):
        return value.item()

    return value


def source_metadata(row: pd.Series) -> dict:
    excluded = {"_selection_index"}
    output: dict = {}

    for key, value in row.items():
        if key in excluded:
            continue

        output[f"normal_{key}"] = metadata_value(value)

    return output


def write_metadata(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise ValueError("No metadata rows were generated.")

    fieldnames = sorted({key for row in rows for key in row})

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[WRITE] {path} rows={len(rows):,}")


# ============================================================
# GENERATION
# ============================================================


def subtype_filename(
    subtype: str,
    index: int,
) -> str:
    short = subtype.replace("q84_s2_", "")

    return f"s2_{short}_{index:05d}.sql"


def generate_subtype(
    con: duckdb.DuckDBPyConnection,
    subtype: str,
    sources: pd.DataFrame,
    metadata_rows: list[dict],
) -> None:
    info = S2_INFO[subtype]

    print(f"\n[GENERATE] {subtype} n={len(sources):,}")

    for index, row in sources.iterrows():
        sql = render_sql(
            subtype=subtype,
            row=row,
            with_provenance=True,
            apply_limit=True,
        )

        validate_structure(subtype, sql)

        # These checks never trigger resampling. A failure aborts the run.
        limited_nrows = count_rows(
            con,
            subtype,
            row,
            apply_limit=True,
        )
        unbounded_nrows = count_rows(
            con,
            subtype,
            row,
            apply_limit=False,
        )

        if limited_nrows <= 0:
            raise RuntimeError(
                f"{subtype}: preselected source {row['file']} "
                "produced an empty result. The run is aborted; "
                "the source is not replaced."
            )

        filename = subtype_filename(
            subtype,
            index,
        )

        (OUT_S2 / filename).write_text(
            sql.strip().rstrip(";") + ";\n",
            encoding="utf-8",
        )

        metadata_rows.append(
            {
                "file": filename,
                "label": int(info["label"]),
                "anomaly_class": info["anomaly_class"],
                "scenario": subtype,
                "category": info["category"],
                "logical_unit": info["logical_unit"],
                "description": info["description"],
                "source_normal_file": str(row["file"]),
                "source_selection_index": int(row["_selection_index"]),
                "n_rows": int(limited_nrows),
                "n_rows_unbounded": int(unbounded_nrows),
                "limit_active": int(unbounded_nrows > limited_nrows),
                **source_metadata(row),
            }
        )

        if (index + 1) % 100 == 0:
            print(f"  [GEN] {subtype}: {index + 1}/{len(sources)}")

    print(f"  [OK] {subtype}: {len(sources):,} queries")


# ============================================================
# MAIN
# ============================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Q84 S2 SQL workloads.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", "--db-path", type=Path, required=True, help="Existing DuckDB database.")
    p.add_argument("--output-dir", type=Path, required=True, help="Parent query directory, e.g. Queries/Q84.")
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
    for name in ['q84_s2_anomalies']:
        pool = args.output_dir / name
        if pool.exists() and (not pool.is_dir() or next(pool.iterdir(), None) is not None):
            p.error(f"Output pool must be absent or empty: {pool}. Use a new output directory.")
    return args


def configure_cli() -> None:
    global DB_PATH, DB_SCHEMA, EXTERNAL_BASE_DIR, N_S2_PER_TYPE, OUT_S2, Q84_NORMAL_DIR, SEED
    args = parse_args()
    DB_PATH = str(args.db)
    DB_SCHEMA = args.schema
    SEED = args.seed
    N_S2_PER_TYPE = args.n_per_type
    OUT_S2 = args.output_dir / "q84_s2_anomalies"
    EXTERNAL_BASE_DIR = args.output_dir
    Q84_NORMAL_DIR = args.normal_dir


def main() -> None:
    configure_cli()
    print("=" * 92)
    print("[Q84 FINAL S2 GENERATOR]")
    print("=" * 92)
    print(f"DB_PATH={DB_PATH}")
    print(f"DB_SCHEMA={DB_SCHEMA}")
    print(f"Q84_NORMAL_DIR={Q84_NORMAL_DIR}")
    print(f"OUT_S2={OUT_S2}")
    print(f"SEED={SEED}")
    print(f"N_S2_PER_TYPE={N_S2_PER_TYPE}")
    print(f"S2_SUBTYPES={S2_SUBTYPES}")
    print("=" * 92)

    sources = select_common_normal_sources()

    OUT_S2.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        con.execute("SET search_path = ?", [DB_SCHEMA])
        run_pre_generation_checks(
            con,
            sources,
        )

        metadata_rows: list[dict] = []

        for subtype in S2_SUBTYPES:
            generate_subtype(
                con=con,
                subtype=subtype,
                sources=sources,
                metadata_rows=metadata_rows,
            )

        write_metadata(
            OUT_S2 / "metadata.csv",
            metadata_rows,
        )

    finally:
        con.close()

    total = N_S2_PER_TYPE * len(S2_SUBTYPES)

    print("\n[DONE]")
    print(f"  queries   : {total:,}")
    print(f"  directory : {OUT_S2}")
    print(f"  metadata  : {OUT_S2 / 'metadata.csv'}")


if __name__ == "__main__":
    main()


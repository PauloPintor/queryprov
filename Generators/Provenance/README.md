# Provenance materialisers

These programs execute existing provenance-annotated SQL against an annotated
DuckDB database. They do not generate SQL workloads, create splits, or evaluate
detectors. Keep all four Python files together in `Generators/Provenance`.

| Program                    | Stored representations                   | Dependency/order                                     |
| -------------------------- | ---------------------------------------- | ---------------------------------------------------- |
| `materialize_full.py`      | Complete Lineage/Atomic membership sets  | Existing SQL and annotated database                  |
| `materialize_full_bags.py` | Complete Token TF/Witness TF, no PSD     | Existing SQL and annotated database                  |
| `materialize_compact.py`   | KMV and membership CountSketch, 512/1024 | Existing SQL and annotated database                  |
| `materialize_bags_psd.py`  | Frequency CountSketch, 512/1024, and PSD | Existing compact batches, SQL and annotated database |

## Environment

Linux or macOS, Python 3.10+, and:

```bash
python -m pip install duckdb numpy pyarrow
```

Use the original database and dependency versions when reproducing an existing
experiment. This package does not establish a new pinned experimental
environment. SQL and metadata alone cannot produce the required provenance.
All database connections are read-only. The scripts use `fcntl` file locks.

## Input layout

`--queries-root` is the directory containing the query directories:

```text
Queries/
  Q3/
    q3_normal/
    q3_anomalies/
    q3_s2_anomalies/
  Q5/
    ...
```

Retain SQL and metadata in their original pool directories. Hidden `._*` SQL
files are ignored. `--queries` explicitly selects numeric query IDs; `--pools`
selects `normal`, `s1` and/or `s2` (all three by default). `--schema` defaults
to `tpcds`. Use `--help` on each script for its full interface.

## 1. Complete membership: sets and binary vectors

```bash
python Generators/Provenance/materialize_full.py \
  --db /home/paulo/Provenance/tpcds_prov.duckdb \
  --queries-root /home/paulo/Provenance/Queries \
  --output-root /home/paulo/Provenance/Queries_Full \
  --queries 3 5 18 20 28 60 66 82 84 85 89 92 93 94 98 \
  --pools normal s1 s2 \
  --representation both \
  --max-witness-combinations 1000000 --timeout-seconds 3600 \
  --threads 2 --memory-limit 8GB
```

Writes `lineage_hash` and `monoids_hash`, plus execution identity and provenance
mode. Both set and binary-vector evaluators must use these same columns.
Vectorisation assigns one coordinate per distinct stored identifier; no
feature hashing, dimensionality reduction or CountSketch is performed here.
Underlying element IDs retain the original 64-bit BLAKE2b encoding. Detected
within-execution collisions fail; these IDs are not a mathematically
collision-free dictionary across all executions.

`--representation lineage` stores only Lineage, without expanding factor
products. `--representation atomic` stores only Atomic Why. Use separate output
roots for different representation modes. Missing columns are omitted, not
replaced by empty sets. No witness lists, bag frequencies or sketches are stored.

## 2. Complete frequency vectors: bags, without PSD

```bash
python Generators/Provenance/materialize_full_bags.py \
  --db /home/paulo/Provenance/tpcds_prov.duckdb \
  --queries-root /home/paulo/Provenance/Queries \
  --output-root /home/paulo/Provenance/Queries_Full_Bags \
  --queries 3 5 18 20 28 60 66 82 84 85 89 92 93 94 98 \
  --pools normal s1 s2 \
  --representation both \
  --max-witness-combinations 1000000 --timeout-seconds 3600 \
  --threads 2 --memory-limit 8GB
```

Writes the full frequency columns expected by the bag-vector evaluator:

- `token_tf`: `list<struct<token: int64, tf: int64>>`;
- `witness_tf`: `list<struct<witness_hash: int64, tf: int64>>`.

Tokens occur once within each canonical witness. Repeated occurrences of that
witness contribute repeatedly to both frequency representations. Frequencies
are stored as exact integers, not normalised values. The old materialiser's
32-bit `tf` storage is widened to 64 bits; field names and frequency semantics
are unchanged. A value beyond int64 causes failure rather than rounding.

There are no PSD columns, result cardinality, KMV or CountSketch in these files.
The program executes SQL; it does not derive full bags from membership sets
or compact sidecars, which do not contain all occurrence information.

`--representation token` stores only Token TF. For supported factor products,
its exact counts are computed without Cartesian enumeration: the frequency of
a token is the number of combinations minus the combinations excluding it.
This also handles tokens shared by several factors, counting them once within
each witness occurrence. `--representation witness` stores only Witness TF.
Use another output root when changing modes.

## Full materialisation: limits and supported formats

**The one-million-combination and one-hour values in the examples are operational
examples, not recovered experimental settings.** They are required explicitly.
The combination limit counts candidate witness occurrences before deduplication,
cumulatively per execution, and is conservative. It is not a sketch budget.

For Atomic/Witness TF, every product is counted before expansion. Oversized
products are refused, never sampled or truncated. Lineage/Token-only modes
avoid Cartesian enumeration and do not apply that expansion limit, but remain
subject to time and resource limits. A full run does not use the compact r=14
selection, even for Q28.

`--layout auto` uses the existing workload conventions:

- Q28: `product`, factors of alternatives;
- Q92/Q94: `sum-products`, a sum of factorised products;
- other queries: `witnesses`, explicitly enumerated witnesses.

For another query using one of these formats, specify `--layout product`,
`--layout sum-products` or `--layout witnesses`. The expansion protection is
generic, not restricted to Q28. Unknown nested structures are rejected;
arbitrary SQL/symbolic provenance formats are not automatically supported.
Run groups requiring different explicit layouts separately. Empty product
factors give no witness; sum-products retains the existing Q92/Q94 optional
factor convention.

The full programs use one reusable spawned worker and one SQL at a time.
`--timeout-seconds` covers SQL/collection and separately bounds worker startup.
The supervisor terminates workers that exceed it. `--memory-limit` only limits
DuckDB, not Python. Optional `--address-space-gb` caps worker virtual address
space using the OS; it is not an RSS cap and must allow library mappings.
The parent buffers up to `--batch-size` records (default 25); decrease it for
large outputs. There is no guaranteed fixed total RAM ceiling.

## 3. KMV and membership CountSketch

```bash
python Generators/Provenance/materialize_compact.py \
  --db /home/paulo/Provenance/tpcds_prov.duckdb \
  --queries-root /home/paulo/Provenance/Queries \
  --queries 3 5 18 20 28 60 66 82 84 85 89 92 93 94 98 \
  --pools normal s1 s2 \
  --workers 4 --threads 2 --memory-per-worker 8GB
```

Writes `batch_XXXX.parquet` beside the SQL, 25 executions per batch, with
Lineage/Atomic KMV-512/1024 and membership CountSketch-512/1024. CountSketch
vectors are raw, not L2-normalised. No full provenance or bag vectors are stored.

**There is no separate output-root option for this program.** Existing main
batches may be replaced under the retained validation/resume rules. Use a
separate SQL dataset tree if that location contains full data to preserve.
Existing incompatible run manifests are rejected. SQL/metadata/splits are not
rewritten. Q28 uses r=14 reduced witnesses when required; Atomic sketches then
describe the selected population, not the full Cartesian population.

## 4. Frequency CountSketch and PSD sidecars

Run after compact materialisation finishes:

```bash
python Generators/Provenance/materialize_bags_psd.py \
  --db /home/paulo/Provenance/tpcds_prov.duckdb \
  --queries-root /home/paulo/Provenance/Queries \
  --queries 3 5 18 20 28 60 66 82 84 85 89 92 93 94 98 \
  --pools normal s1 s2 \
  --workers 4 --threads 2 --memory-per-worker 8GB
```

Reads main-batch identities and re-executes SQL. It verifies recomputed membership
summaries against the main batches before writing
`batch_XXXX.bagcompact.parquet`. Main batches are not modified. Incompatible
sidecars are rejected, not silently replaced. Nondeterministic SQL selections
can lead to a mismatch and refusal.

Sidecars contain Token TF/Witness TF CountSketch-512/1024 and PSD counts/ratios,
including result cardinality. For Q28, Token TF and PSD counts use factor
arithmetic; Witness TF uses selected witnesses when expansion is reduced.
Full/selected scopes are recorded separately. Exact large counts also have
decimal-string copies; float64 PSD values may round integers above 2\*\*53.

The compact programs retain their hashes, seeds, dimensions, r=14 strategy and
batch sizes. `--memory-per-worker` limits DuckDB, not the whole process; Python
sets/arrays require additional RAM. They share a lock and cannot run together
on the same root. Temporary spill files use the root's parent directory.

## Identities, completion and resume

The full programs require a separate output tree, outside the input tree, and
never write the input. SQL and metadata are not copied. Point evaluators at
the chosen output tree. Execution IDs are reused from source Parquets or
`compact_identities.csv`; otherwise deterministic IDs are assigned by sorted
SQL name. Existing splits then require identity/name validation, not an
assumption of numeric-ID equivalence. No program creates or modifies splits.

For full runs, the first failing execution stops its pool; processing then
continues with the next pool. Uncommitted batches are discarded, completed
batches retained, and no success marker is written for the failed pool.
Inspect `full_errors.json` and `full_status.json`. Worker exits are not assumed
to be OOM without OS evidence. Input/manifest errors can abort the entire run.

Repeat the identical command to resume. Full run input/database/configuration/
code changes require a new output root. Completed batches are checked and
reused; they do not trigger SQL execution. Non-deterministic SQL can still
change when an execution must actually be rerun. Compact runs retain their
original fingerprint and compatibility checks; code changes may affect resume.

Completion markers are `_SUCCESS.json` for full/compact main outputs and
`_BAG_PSD_SUCCESS.json` for compact bag/PSD sidecars. Do not evaluate partial
pools based only on the existence of some batch files.

## Validation performed

Compact scripts passed their existing self-tests and CLI checks. Full scripts
were tested on synthetic data for membership, duplicates, factor expansion,
expansion refusal, vector roundtrip, real DuckDB execution, spawned workers,
resume and timeout. Full bags were additionally checked against exhaustive
enumeration on 100 random small products, including repeated alternatives and
tokens shared across factors, and checked for absence of PSD columns.
These checks do not replace validation against all real workload SQL files or
the original experimental database.

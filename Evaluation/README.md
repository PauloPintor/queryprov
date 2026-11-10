# Evaluation

Seven public entry points evaluate S1 or S2 using existing shared partitions:

```text
Evaluation/
  evaluate_sets_full.py
  evaluate_sets_kmv.py
  evaluate_vectors_full.py
  evaluate_vectors_countsketch.py
  evaluate_bags_full.py
  evaluate_bags_countsketch.py
  evaluate_psd.py
  _set_evaluation/
    __init__.py
    cli.py
    full_s1.py
    full_s2.py
    kmv.py
    vec_full_s1.py
    vec_full_s2.py
    vec_countsketch.py
    countsketch_cli.py
    bag_full_s1.py
    bag_full_s2.py
    bag_full_adapter.py
    bag_countsketch.py
    psd.py
    psd_cli.py
  README.md
```

Keep `_set_evaluation` beside the entry points. It contains the original
scientific implementations and the shared command-line orchestration; do not
execute its files directly. PSD evaluation is separate from the bag evaluators.

## Requirements

Linux or macOS, Python 3.10+, and:

```bash
python -m pip install numpy pyarrow numba scipy scikit-learn threadpoolctl
```

These evaluators read stored Parquets: DuckDB and SQL execution are not required.
For reproducing experiments, retain their original dependency versions. This
initial package does not claim a pinned reproduction environment.

## Required inputs

Data must follow the existing query/pool structure:

```text
Queries/Q3/q3_normal/*.parquet
Queries/Q3/q3_anomalies/*.parquet       # S1
Queries/Q3/q3_s2_anomalies/*.parquet    # S2
```

The existing flattened pool layout and spelling aliases in the full/set source
evaluators remain supported. CountSketch expects the nested structure above.
Hidden `._*` Parquets are excluded. Full evaluators exclude derived bag sidecars;
the bag CountSketch evaluator reads the paired `.bagcompact.parquet` sidecars.

Select the appropriate source:

- Full: Parquets containing `lineage_hash` and/or `monoids_hash`.
- KMV: Parquets containing the stored Lineage/Atomic KMV columns for the
  requested budget; existing alternative column names remain supported.
- Full bags: Parquets containing `token_tf` and/or `witness_tf` (see below).
- CountSketch: stored membership vectors, or frequency-weighted vectors in
  paired bag sidecars. These require their original materialisation contracts.

Both require `query_id` and `query_name`. A new-materialiser pool with a run
manifest but no `_SUCCESS.json` is rejected as incomplete. Legacy datasets
without materialisation manifests are accepted subject to schema and split
checks; absence of a marker alone is not a full data integrity audit.

Existing split filenames must be:

```text
# S1, in the selected --splits-dir
q3_5k_shared_split_ids.csv

# S2, in the selected --splits-dir
q3_5k_s2_shared_split_ids.csv
```

The names similarly apply to other queries and scales. Pass the actual folder
path; `Splits_S2` and `Splits_s2` are different names on case-sensitive systems.

Manifest columns include `query`, `scale`, `split`, `pos`, and `identity`.
Identity is `numeric_id|query_name`; matching is exact. Changed IDs are not
silently remapped, even if the name matches. Missing, overlapping, duplicate,
wrong-scenario or incompatible partitions fail. The two CountSketch evaluators
retain the strictly checked legacy-ID translations described in their sections.
No evaluator creates,
recreates, resamples, nor modifies splits. Old environment flags cannot enable
split creation in these versions.

## Validate before scoring

```bash
python Evaluation/evaluate_sets_full.py \
  --queries-root /home/paulo/Provenance/Queries_Full \
  --splits-dir /home/paulo/Provenance/Splits_S2 \
  --output-dir /home/paulo/Provenance/Results_Set_Full_S2 \
  --scenario s2 --queries 3 5 18 \
  --scales 5k 10k 25k 50k \
  --representations lineage atomic --neighbors 1 \
  --check-splits
```

`--check-splits` reads identities and validates the existing partitions without
scoring or creating result files. For KMV, use the KMV entry point and also
specify `--kmv-k`. Remove `--check-splits` to evaluate.

## Complete sets

From the repository root:

```bash
python Evaluation/evaluate_sets_full.py \
  --queries-root /home/paulo/Provenance/Queries_Full \
  --splits-dir /home/paulo/Provenance/Splits \
  --output-dir /home/paulo/Provenance/Results_Set_Full_S1 \
  --scenario s1 \
  --queries 3 5 18 20 60 82 84 85 89 92 93 94 98 \
  --scales 5k 10k 25k 50k \
  --representations lineage atomic \
  --neighbors 1 --timeout-seconds 18000
```

Select only queries/scales/representations with the required completed inputs
and existing splits. The query list above is an example, not a claim that all
those full configurations completed. To evaluate S2, change `--scenario`,
`--splits-dir` and `--output-dir`. The same two representation names apply.

## KMV summaries

```bash
python Evaluation/evaluate_sets_kmv.py \
  --queries-root /home/paulo/Provenance/Queries \
  --splits-dir /home/paulo/Provenance/Splits_S2 \
  --output-dir /home/paulo/Provenance/Results_Set_KMV_S2 \
  --scenario s2 \
  --queries 3 5 18 20 28 60 66 82 84 85 89 92 93 94 98 \
  --scales 5k 10k 25k 50k \
  --representations lineage atomic \
  --kmv-k 512 1024 \
  --neighbors 1 --timeout-seconds 18000
```

Use `--kmv-k 512` or `--kmv-k 1024` for a single budget. The evaluator compares
the stored summaries; it does not construct them, resample witnesses, or turn
the reduced Q28 witness population into a full one.

## Protocol retained

The following set-specific details apply to the full/KMV set entry points.
Vector-specific commands and differences appear below.

- Exact all-fitting-execution comparison; no candidate selection.
- Original exact Jaccard for full sets and original KMV Jaccard estimator.
- `--neighbors 1` or `--neighbors 5` is required explicitly. The examples use
  1-NN; the supplied S1/KMV scripts had 5-NN configured. For k=5 the original
  score uses one minus the mean of the five largest similarities.
- Random seed 13, 80/20 fitting/calibration division encoded in existing splits.
- Threshold percentiles 95 and 99, with 1000 bootstrap resamples and the original
  mean-percentile threshold and confidence interval calculation.
- Original empty-provenance score 0.5, metrics, numerical dtypes and read retries.

The four supported training scales retain the original sizes:

| Scale | Train history | Normal test | Anomaly test |
| --- | ---: | ---: | ---: |
| 5k | 5000 | 1250 | 66 |
| 10k | 10000 | 2500 | 125 |
| 25k | 25000 | 6250 | 313 |
| 50k | 50000 | 12500 | 625 |

Scales are processed in increasing size. No query is silently excluded by a
hard-coded list; `--queries` is mandatory.

The supplied full S1 engine loads selected rows in original Parquet order;
the full S2 engine uses manifest order. The KMV engine retains its original
Parquet ordering. These differences were intentionally preserved because
reordering calibration scores can change seeded bootstrap samples. Audit
files record the loading policy; the stored manifest identity lists do not
claim to be score-array order when the policy is `parquet`.

## Resources and failures

Each configuration runs in an isolated subprocess. `--timeout-seconds` defaults
to 18000 (five hours), including worker startup and Numba warm-up. The worker
is killed on timeout. `--threads` optionally sets Numba parallelism; if omitted,
the runtime's normal default is retained. It does not cap total RSS or control
every Arrow/BLAS thread.

After a timeout, Python `MemoryError`, or a worker terminated with SIGKILL,
larger scales are skipped for that query/representation/budget. Other queries
and representations continue. Previously completed larger-scale results remain
available. SIGKILL is labelled `oom_or_sigkill`, not claimed to prove OOM.
Other failures are reported and do not trigger this resource-based skip rule.

Missing splits/schema problems fail preflight before scoring. No assumption
that the input is mounted at `/mnt/provenance_ro` remains; data are only read.
Output must be a separate directory outside both data and split trees.
Temporary files and Numba cache are under the selected output directory.

## Outputs and resume

Existing per-configuration CSV naming and scientific result fields are retained.
Output also includes:

- Combined result/error CSVs with the original S1/S2 names.
- `WorkerResults/`: result and task JSON files.
- `Logs/`: complete worker output.
- `SplitIdentityAudit/`: manifest identities, hashes and loading policy.
- `Scores/`: the KMV engine's existing compressed score arrays. The original
  full engines did not export score arrays; this adaptation does not add them.
- `evaluation_run.json`: explicit configuration, source code hashes, split
  checksums and Parquet size/modification-time inventory.
- `evaluation_status.json`: overall completion and result/error counts.

Successful metric CSVs are supplemented with scenario, k, split hashes and job
signature. Old metadata strings that said `1nn` unconditionally now report the
selected k; the score calculation itself is unchanged. End messages include
the local date/time of completion.

Repeat the identical command to resume. Cached successful configurations are
reused. `--rerun-completed` reruns them; `--rerun-resource-failures` retries
resource failures. Changed inputs, configuration, selection or code require a
new output directory to prevent mixing runs. Legacy output directories without
this package's run manifest must not be used for resume. Parquet fingerprints
use size/mtime, not full content hashing. Do not modify input files during a run.

Only one evaluator may write a given output directory at once. Use separate
directories for S1/S2, different k values or different configuration selections.
A nonzero exit status indicates failure or skipped configurations, not a
complete evaluation.

## Validation scope

For the set evaluators, the extracted scientific functions/classes are
structurally identical to the three supplied set sources. Small-set numerical
tests check 1-NN, 5-NN and bootstrap
calculations. Synthetic 5k datasets exercise both representations, both
scenarios, subprocess execution, existing split reuse and resume. These tests
do not rerun the original research dataset or establish identical timing across
machines and dependency versions.

## Full binary vectors, without bags

```bash
python Evaluation/evaluate_vectors_full.py \
  --queries-root /home/paulo/Provenance/Queries_Full \
  --splits-dir /home/paulo/Provenance/Splits \
  --output-dir /home/paulo/Provenance/Results_Vector_Full_S1_1NN \
  --scenario s1 --queries 3 5 18 \
  --scales 5k 10k 25k 50k \
  --representations lineage atomic \
  --neighbors 1 --timeout-seconds 18000 \
  --jobs -1 --working-memory-mb 512
```

For S2, change `--scenario s2`, the split directory, and the output directory.
For 5-NN, use `--neighbors 5` and a different output directory. Add
`--check-splits` to validate without scoring.

The inputs are exactly the full set columns `lineage_hash`/`monoids_hash`.
Each distinct selected identifier has one binary coordinate. The vocabulary
includes fitting, calibration, and both test partitions so that identifiers
first encountered outside fitting are not dropped. This coordinate mapping
does not learn from labels. No bags, SVD, CountSketch, dimensionality reduction
or fit-only feature filtering is introduced.

`--jobs` controls sklearn pairwise parallelism. `--working-memory-mb` controls
the pairwise distance chunk target, not total process RSS. These vector full
entry points do not use the set-specific `--threads` option. The supplied full
vector engines both retain their original Parquet row order, even under S2.

Successful outputs retain the source `method=knn_distance` and features
`vec_lineage`/`vec_atomic`, CSVs, vocabulary audit, score arrays and execution
identities. Combined results are `full_vector_shared_results.csv` (S1) and
`full_vector_s2_shared_results.csv` (S2). `evaluation_signature` protects the
shared scheduler's checkpoints; the vector `job_signature` in CSV/NPZ includes
the selected k and resource settings. Timeout/resource-failure skips and output
directory isolation follow the set scheduler described above.

## Membership CountSketch vectors, without bags

```bash
python Evaluation/evaluate_vectors_countsketch.py \
  --queries-root /home/paulo/Provenance/Queries \
  --splits-dir /home/paulo/Provenance/Splits_S2 \
  --output-dir /home/paulo/Provenance/Results_CountSketch_S2_1NN \
  --scenario s2 --queries 3 5 18 \
  --scales 5k 10k 25k 50k \
  --representations lineage atomic \
  --dimensions 512 1024 --neighbors 1 \
  --threads 4 --score-batch-rows 256 \
  --max-matrix-gib 4 --timeout-seconds 18000
```

Use `--dimensions 512` or `1024` to evaluate only one size. Use
`--neighbors 5` in another output directory for 5-NN. `--check-splits` runs
the original input/identity preflight without scoring. Only existing splits
are used; no resampling or split creation is possible.

The original pipeline is retained: raw stored membership CountSketch -> L2
normalisation -> exhaustive cosine neighbours against fitting executions.
The input must include both the requested vector columns and the original
provenance/hash contract columns. `_SUCCESS.json` with matching batch/row counts
is required. Bag sidecars are neither required nor read. Stored Q28 reduced
witness populations remain unchanged; the evaluator performs no new sampling.

`--threads` controls BLAS, and `--max-matrix-gib` retains the original guard on
selected matrices; neither is a total memory cap. This script retains the
CountSketch source's independent-job scheduling: failure of one scale does not
automatically suppress larger scales. Workers have the configured timeout.

The supplied CountSketch code includes legacy numeric-ID translation checks.
They are retained, not generalised: only the known offsets are accepted, with
unique names, matching subtypes and verified current SQL hashes. All other
identity mismatches fail. Translations are audited; original split files and
their membership/order are not changed. Therefore SQL files are required when
such a translation is necessary, even though no SQL is executed.

Output retains the original S1/S2 subdirectories and aggregate naming:
`S1/countsketch_s1_results_all.csv` or `S2/countsketch_s2_results_all.csv`, plus
the root aggregate, score NPZs, errors, logs and `InputAudit`. Resuming verifies
the saved identities, labels, AP/AUROC and score-file checksums. Configuration
changes, including k, require another output directory. Native job signatures
include the selected neighbourhood size. This entry point does not expose the
set scheduler's rerun flags; successful verified jobs are reused.

## Neighbourhood size in vectors

Both vector entry points accept 1-NN and 5-NN. The supplied full S1 script had
`K_NEIGHBORS=5` in its configuration but rejected any value other than 1 in its
scoring function. This version explicitly implements both options:

- 1-NN: the original nearest-distance path is retained.
- 5-NN: mean distance to the five nearest fitting executions. Equivalently,
  one minus the mean of the five largest cosine similarities for normalised
  CountSketch vectors, following the same aggregation principle as the sets.

If fewer than k fitting executions are available, k is bounded by their count,
as in the set scoring functions. Production splits contain many more than five.
No fitting/calibration/test selection or bootstrap rule changes with k.

The original empty-vector conventions remain distinct: full binary evaluation
uses 0.5 for an empty evaluated set; CountSketch uses zero cosine similarity
for a zero vector, giving distance 1. Neither policy was silently harmonised.
The CountSketch clipping range remains [0, 2], appropriate for signed vectors.

Vector tests check lossless membership and unseen identifiers, 1-NN/5-NN against
explicit cosine distances, signed sketches, and the retained empty-vector
policies. Synthetic 5k integrations cover both scenarios, both representations,
both k values, score export and resume. No full research experiment was rerun.

## Full bags: Token TF and Witness TF

```bash
python Evaluation/evaluate_bags_full.py \
  --queries-root /path/to/Queries_Full_Bags \
  --splits-dir /path/to/Splits \
  --output-dir /path/to/Results_Bags_Full_S1_1NN \
  --scenario s1 --queries 3 5 18 \
  --scales 5k 10k 25k 50k \
  --representations token witness \
  --neighbors 1 --timeout-seconds 18000 \
  --jobs -1 --working-memory-mb 512
```

For S2, select `--scenario s2` and the S2 split/output directories. For 5-NN,
select `--neighbors 5` and a separate output directory. `--check-splits`,
`--rerun-completed` and `--rerun-resource-failures` follow the full/set scheduler.
After timeout, MemoryError or SIGKILL, larger scales are skipped only for that
query and representation; processing continues for the other representation.

Input columns are `token_tf: list<struct<token, tf>>` and
`witness_tf: list<struct<witness_hash, tf>>`, with the original metadata
`query_id`, `template_id`, `query_name`, `subtype`. These are complete stored
frequencies, not membership sketches or `.bagcompact.parquet` sidecars.
Full bag materialisations from `materialize_full_bags.py` are supported.

The original sparse frequency construction is retained. Vocabulary covers all
selected fitting, calibration and test executions without using their labels;
frequencies are not converted to binary membership. No TF-IDF, SVD, feature
hashing, PSD standardisation or reduction is applied. Cosine nearest-neighbour
search uses only fitting executions as references. The source's float32
frequency matrices and cosine handling of zero rows remain unchanged.
An entirely empty vocabulary remains an error, as in the supplied code.
Rows are scored in manifest order, as in both supplied bag engines.

The original full bag 5-NN code returned distance to the fifth neighbour.
Here, at the user's request to follow the sets, 5-NN returns the **mean distance
to the five nearest fitting executions**. 1-NN retains its original distance.
This is an explicit scoring change for k=5; legacy 5-NN outputs must not be
treated as equivalent or reused. No split or calibration rule was changed.

Outputs include `all_results.csv`, `all_errors.csv`, per-configuration CSVs,
worker logs/checkpoints and split identity audits. Score NPZs additionally retain
calibration/test scores, labels, fit/calibration/test identities and test
subtypes for later comparisons. PSD/scalar and unused LOF code was removed
from these bag engines. Only Token TF and Witness TF are selectable.

## Bag CountSketch: Token TF and Witness TF

```bash
python Evaluation/evaluate_bags_countsketch.py \
  --queries-root /path/to/Queries_Compact \
  --splits-dir /path/to/Splits_S2 \
  --output-dir /path/to/Results_Bags_CountSketch_S2_1NN \
  --scenario s2 --queries 3 5 18 \
  --scales 5k 10k 25k 50k \
  --representations token witness \
  --dimensions 512 1024 --neighbors 1 \
  --threads 4 --score-batch-rows 256 \
  --max-matrix-gib 4 --timeout-seconds 18000
```

Both scenarios and k values are supported with the same switches as above.
Use `--dimensions 512` or `1024` to select a single dimension. The source's
independent-job scheduling is retained, as for membership CountSketch.

This reads `bag_lineage_countsketch_{dimension}` (Token TF) and
`bag_atomic_countsketch_{dimension}` (Witness TF) from paired
`batch_NNNN.bagcompact.parquet` files. It retains the source's stable
max-absolute-value rescaling followed by L2 normalisation in float64 and cosine
scoring in float32. It does not reconstruct witnesses or generate new sketches.
1-NN keeps the original scoring path; 5-NN averages the five nearest distances.
Zero-vector distance remains 1 and signed cosine distances remain in [0, 2].

The original main/sidecar binding, SQL identity, population, hash, baseline and
success-marker checks remain mandatory. In particular, `_SUCCESS.json` and
`_BAG_PSD_SUCCESS.json` are required. The historical names `BAG_PSD_VERSION`
and `bag_psd_*` refer to the existing sidecar format; they are not PSD features.
Population counters are read to verify the stored bag construction, **not to
evaluate PSD**. Removing these format checks would break compatibility with
the supplied sidecars. No PSD vectors, scores or results are produced.

Known legacy-ID offsets require the same unique-name/subtype/SQL-hash checks
as in the supplied bag evaluator. Split files are never rewritten. Existing
Q28 population restrictions are retained, not generalised by this evaluator.

Results include `S1/bag_countsketch_s1_results_all.csv` or
`S2/bag_countsketch_s2_results_all.csv`, plus root aggregates, score NPZs,
input audits and logs. Resume validates stored results and score checksums.
Use a new output directory for a different scenario, k, dimension selection
or code version. `--check-splits` performs the original input preflight without
scoring. The CountSketch entry points do not expose full-scheduler rerun flags.

Bag validation covers frequency preservation, all-split vocabulary, cosine
1-NN/5-NN, empty rows, and synthetic 5k CLI runs with both representations,
S1/S2, 512/1024 dimensions, score export and resume. It does not replace a run
on the original research dataset.

## PSD and PSD+R

One entry point replaces `Eval_Scalars.py` and `Eval_Scalars_NoCount.py`:

```bash
python Evaluation/evaluate_psd.py \
  --queries-root /path/to/Queries_Compact \
  --splits-dir /path/to/Splits \
  --output-dir /path/to/Results_PSD_S1_1NN \
  --scenario s1 --queries 3 5 18 \
  --scales 5k 10k 25k 50k \
  --variant psd --neighbors 1 \
  --jobs -1 --working-memory-mb 512 --timeout-seconds 18000
```

- `--variant psd`: seven original provenance descriptors, excluding `nrows`.
- `--variant psd-r`: the same seven descriptors, with `nrows` prepended (eight).

Use a separate output directory for each variant, scenario or k. S2 uses
`--scenario s2` and the S2 split directory. Both `--neighbors 1` and
`--neighbors 5` are supported. Queries and all four training sizes are selected
explicitly rather than inherited from the commented source settings.

The seven coordinates, in original order, are distinct witnesses, total
witness occurrences, distinct tokens, total token occurrences, token
unique/total ratio, witness unique/total ratio, and distinct witnesses/tokens
ratio. Original denominator handling and float32 conversion are retained.
The supplied `Eval_Scalars.py` had `nrows` commented out while still requiring
eight coordinates; the explicit variant selection fixes that inconsistency.
Seven-feature results use the valid `Eval_Scalars_NoCount.py` feature mapping.

Inputs are the five scalar count columns already stored in
`.bagcompact.parquet` sidecars. Both variants read and validate the existing
eight-feature format; only the evaluated projection differs. Paired main files
and `_BAG_PSD_SUCCESS.json` are required by the source checks. Sketch vectors
are not read. Stored population metadata, including Q28, is preserved and
exported, not recomputed or replaced. No SQL is executed.

The pipeline remains stored counts/ratios -> **StandardScaler fitted only on
the fitting partition** -> cosine nearest neighbours. Calibration and test
rows use that fitted scaler. This is numerical scaling, not the fit-only
vocabulary mapping removed from the full sparse vector evaluators.
Bootstrap calibration, percentiles, labels and shared split membership/order
are unchanged. Only k-NN is exposed; unused LOF code is excluded.

As in the bag full adaptation, the source's fifth-neighbour distance has been
replaced for k=5 by the mean of the five nearest distances, following the sets.
The original 1-NN path is unchanged. Do not mix old fifth-distance results with
these mean-distance results.

The source's unique-query-name ID translation within a pool is retained and
audited, with subtype matching. Unlike the CountSketch scripts, the scalar
source does not require a known numeric offset or SQL hash for translation.
Duplicate/missing names fail; split identities and files are not rewritten.
`--check-splits` verifies selected identities and scalar feature mapping without
scoring. It reads selected sidecar counts, so it can take time on large pools.

Outputs preserve `all_results.csv`, `all_errors.csv`, scenario-specific
`S1/scalars_s1_results_all.csv` or `S2/scalars_s2_results_all.csv`, worker logs,
split audits, and score NPZs. NPZs retain raw feature matrices, feature names,
fitted scaler statistics, identities, labels and scores. CSVs identify
`psd_variant`, feature dimension, k and population metadata.

Resume verifies configuration/input/code signatures and saved score checksums.
`--rerun-completed` reruns completed jobs; `--rerun-resource-failures` retries
resource failures. Existing incompatible results require a new output folder.
The source's independent-job scheduling is retained: one failed scale does
not suppress larger scales. Failed or cached-failed configurations produce a
nonzero final exit status rather than a false success.

Tests cover both dimensions, cardinality inclusion/exclusion, descriptor
formulas, cosine 1-NN/5-NN, fitting-only scaler statistics, and synthetic 5k
CLI runs for both scenarios with score export and resume. Full research
experiments were not rerun.

# Beyond Query Syntax and Results: Data Provenance for Detecting Unusual Database Executions

This repository accompanies the article by **Paulo Pintor, Rogério Luís Costa,
and José Moreira**. It provides a labelled SQL workload and a Python pipeline
for investigating whether data provenance can distinguish normal from unusual
database query executions, even when queries remain valid and executable.

Query syntax and result-level measurements may not fully capture these
deviations. Data provenance provides complementary evidence by recording which
input tuples contributed and how they were combined. The study evaluates this
information through complete representations, KMV and CountSketch summaries,
and structural descriptors.

## Contents

- [Workload and repository contents](#workload-and-repository-contents)
- [Glossary](#glossary)
- [Installation](#installation)
- [Quick start: Q3, S1, KMV-512](#quick-start-q3-s1-kmv-512)
- [Other representations](#other-representations)
- [Experimental protocol](#experimental-protocol)
- [Witness expansion and sensitivity](#witness-expansion-and-sensitivity)
- [Citation and license](#citation-and-license)

## Workload and repository contents

The workload uses TPC-DS at scale factor 1 and comprises 15 query templates:
**Q3, Q5, Q18, Q20, Q28, Q60, Q66, Q82, Q84, Q85, Q89, Q92, Q93, Q94,
and Q98**. Each template has its own normal execution history and two anomaly
scenarios:

| Scenario                     | Description                                                                                                       |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **S1: parameter anomalies**  | Parameter values depart from the normal parameter profiles.                                                       |
| **S2: structural anomalies** | Relational structure changes through applicable join, predicate, aggregation, limit, or subquery-block mutations. |

Labels identify controlled deviations, not necessarily malicious activity.
The unit of analysis is an individual query execution, rather than a user or
session. Structural mutation categories depend on the operations present in
each template.

The repository includes generated SQLs and metadata, workload generators,
shared evaluation partitions, database annotation scripts, provenance
materialisers, evaluators, and a witness-expansion sensitivity pilot.
**The TPC-DS database, materialised Parquets, and experimental results are not
bundled.** These are prepared or generated locally.

```text
queryprov/
  Queries/                   # Normal, S1 and S2 SQLs and metadata
  Splits_S1/                 # Shared S1 partitions
  Splits_S2/                 # Shared S2 partitions
  Generators/
    Database/                # Provenance token preparation
    Workload/                # SQL workload generators
    Provenance/              # Full and compact materialisers
  Evaluation/
    _set_evaluation/          # Implementation modules used by the entry points
    Sensitivity/             # Reduced witness-expansion pilot
    evaluate_*.py            # Evaluation entry points
  requirements.txt
  LICENSE
  README.md
```

Use the supplied SQLs directly for the distributed workload. Regeneration
through [`Generators/Workload/`](Generators/Workload/) is optional and should
not be assumed to produce the same executions or identities as the supplied
SQLs and splits.

## Glossary

| Term                      | Meaning in this repository                                                                                                          |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Token**                 | Identifier assigned to an input tuple in its table's `prov` column.                                                                 |
| **Lineage**               | Set of input-tuple tokens contributing to an execution's results.                                                                   |
| **Witness**               | Set of input-tuple tokens participating jointly in a derivation of an output tuple. Distinct derivations may have the same witness. |
| **Atomic Why-provenance** | Execution-level set of witnesses, with each complete witness treated as one indivisible element.                                    |
| **Token TF / Witness TF** | Frequency representations retaining token participation or witness occurrence counts.                                               |
| **PSD / PSD+R**           | Provenance Structural Descriptors: seven counts and ratios. PSD+R adds result cardinality.                                          |
| **KMV**                   | K-minimum-values set sketch, used to estimate Jaccard similarity.                                                                   |
| **CountSketch**           | Signed-hashing projection into a fixed-dimensional vector, used here with cosine distance.                                          |
| **1-NN / 5-NN**           | Distance to the nearest fitting execution, or mean distance to the five nearest fitting executions.                                 |

## Installation

Use Linux or macOS with Python 3.10 or later and compatible package versions.
The scripts use Unix file-locking and resource interfaces; native Windows
execution is not supported.

```bash
git clone https://github.com/PauloPintor/queryprov.git
cd queryprov
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements file lists runtime dependencies rather than a pinned
historical environment. Record the Python and package versions used for each
run. The article's experiments used an Intel Core i7-1265U with 32 GB RAM,
running Arch Linux.

## Quick start: Q3, S1, KMV-512

This example prepares compact Q3 data and evaluates Lineage at the 5k training
size. It requires an existing DuckDB database containing TPC-DS SF1 tables in
the `tpcds` schema. Creating or importing those tables is a prerequisite, not
part of the Python installation.

### 1. Add provenance tokens

Preview the annotation, then apply it to a backed-up database:

```bash
python Generators/Database/add_provenance_tokens.py \
  --database /path/to/tpcds_prov.duckdb --schema tpcds

python Generators/Database/add_provenance_tokens.py \
  --database /path/to/tpcds_prov.duckdb --schema tpcds --apply
```

Existing tokens are not overwritten. Preserve the annotated database when
reusing materialised provenance: the preparation script reconstructs the
prefix-plus-sequence rule, but cannot establish the historical row-to-token
mapping. See the [database guide](Generators/Database/README.md) for validation
and mapping details.

### 2. Materialise compact Q3 pools

Prepare a separate working SQL tree at `/path/to/Queries_Compact`, preserving
the supplied layout, including `Q3/q3_normal/` and `Q3/q3_anomalies/`.
Then run:

```bash
python Generators/Provenance/materialize_compact.py \
  --db /path/to/tpcds_prov.duckdb \
  --queries-root /path/to/Queries_Compact \
  --queries 3 --pools normal s1 \
  --workers 1 --threads 2 --memory-per-worker 4GB
```

**This processes all SQLs in the selected pools, not just 5k executions.**
The training size is selected later by the evaluator. The materialiser writes
main Parquet batches beside the SQLs, so keep this tree separate from any full
Parquets you wish to preserve. Resource values above are example settings.

Wait for successful pool completion before proceeding. The
[materialisation guide](Generators/Provenance/README.md) explains completion
markers, errors, and resume behaviour.

### 3. Validate and evaluate

From the repository root, validate the selected inputs against the supplied
S1 partitions:

```bash
python Evaluation/evaluate_sets_kmv.py \
  --queries-root /path/to/Queries_Compact \
  --splits-dir ./Splits_S1 \
  --output-dir /path/to/Results_KMV_S1_Q3 \
  --scenario s1 --queries 3 --scales 5k \
  --representations lineage --kmv-k 512 --neighbors 1 \
  --check-splits
```

Remove `--check-splits` to evaluate. For S2, materialise the S2 pool and select
`--scenario s2` with `--splits-dir ./Splits_S2` and a different output directory.

Evaluators reuse existing partitions; they never create or resample them.
Do not regenerate splits or renumber executions to bypass identity errors.
Write results outside the input trees, and keep different configurations in
separate output directories. Unchanged commands support resume; changed
inputs, configuration, or code may require a new output directory.

## Other representations

Complete binary vectors retain the membership information of the corresponding
stored sets, without feature hashing or dimensionality reduction. They are
distinct from the summarised CountSketch vectors. Frequency vectors retain
occurrence counts rather than converting them to binary membership.

| Representation                     | Evaluator in `Evaluation/`        | Materialiser in `Generators/Provenance/` |
| ---------------------------------- | --------------------------------- | ---------------------------------------- |
| Complete Lineage / Atomic Why sets | `evaluate_sets_full.py`           | `materialize_full.py`                    |
| Complete binary membership vectors | `evaluate_vectors_full.py`        | `materialize_full.py`                    |
| Complete Token TF / Witness TF     | `evaluate_bags_full.py`           | `materialize_full_bags.py`               |
| Lineage / Atomic Why KMV           | `evaluate_sets_kmv.py`            | `materialize_compact.py`                 |
| Membership CountSketch             | `evaluate_vectors_countsketch.py` | `materialize_compact.py`                 |
| Frequency CountSketch              | `evaluate_bags_countsketch.py`    | Compact, then `materialize_bags_psd.py`  |
| PSD / PSD+R                        | `evaluate_psd.py`                 | Compact, then `materialize_bags_psd.py`  |

The bag/PSD sidecar materialiser requires completed compact main batches.
Full materialisation is a separate branch; it is not required before running
the compact branch. Materialisers execute SQL, while evaluators read the
resulting stored representations.

All evaluation entry points support S1/S2 and depend on the accompanying
`_set_evaluation/` directory. The [evaluation guide](Evaluation/README.md)
provides commands, input formats, identity checks, outputs, and resource
controls. Use each script's `--help` for its complete interface.

## Experimental protocol

The article evaluates **5k, 10k, 25k, and 50k** normal training executions,
partitioned into **80% fitting and 20% calibration**. Separate test sets contain
approximately **5% anomalies**. Representations share the same partitions for
each template, scale, and scenario.

The reported comparisons use **1-NN**, with Jaccard-based scores for sets and
cosine distance for vectors. The released evaluators also support **5-NN** as
defined in the glossary. Normal calibration scores establish thresholds at
the 95th and 99th percentiles, using 1,000 bootstrap samples.

Both summary sizes, 512 and 1024, are available. The article retains
**KMV-512** and **CountSketch-1024** for subsequent summarised comparisons.
These budgets denote hash values and vector coordinates, respectively, not
equal storage allocations.

Evaluation time starts from stored representations and excludes SQL execution
and provenance materialisation. Complete evaluations have a default five-hour
limit per configuration and may also encounter memory limitations. The article
therefore compares common subsets with completed results; requesting every
template does not guarantee completion. Materialisation limits are separate
from the evaluation timeout. Consult the technical guides for measurement
scopes and failure handling.

## Witness expansion and sensitivity

Large products in factorised provenance can make complete witness enumeration
infeasible. Full materialisers refuse expansions beyond their configured
limit. Compact witness construction retains **r=14** alternatives per factor
when reduction is required, recording that the resulting witness summaries
describe the selected population rather than the full Cartesian expansion.

The [Q28 sensitivity pilot](Evaluation/Sensitivity/README.md) compares
**r=14, 15, and 16** on fixed executions, measuring construction cost and
nearest-reference score sensitivity for Atomic KMV and membership CountSketch.
It does not use a complete-witness baseline or evaluate bag/PSD sensitivity.
Expansion rules and population metadata are documented in the materialisation
guide.

## License

The code is distributed under the [MIT License](LICENSE). TPC-DS is an external
dependency and remains subject to its own terms.

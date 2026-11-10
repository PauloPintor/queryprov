# Witness-expansion sensitivity

This is the fixed Q28 pilot used to examine the reduced witness expansion.
It retains the original 28 SQL filenames and roles (8 normal references,
20 evaluation executions), r=14/15/16, both KMV budgets (512/1024), both binary
Atomic CountSketch dimensions (512/1024), seeds and witness hashing.
No replacement SQLs are chosen if an original filename is missing.

From the repository root:

```bash
python Evaluation/Sensitivity/q28_expansion_sensitivity.py \
  --database /path/to/tpcds_prov.duckdb \
  --queries-root /path/to/queryprov/Queries/Q28 \
  --schema tpcds --threads 2 \
  --output-dir /path/to/Q28_Expansion_Sensitivity
```

Run the numerical self-test without a database:

```bash
python Evaluation/Sensitivity/q28_expansion_sensitivity.py --self-test
```

The database is opened read-only. SQLs must contain exactly one SELECT.
Existing SQL, splits and Parquets are never modified. Each invocation writes
a new dated run directory and ZIP, so it does not overwrite an earlier run.
The original exact-product threshold is 14**6; below it, factors are expanded
exactly for every tested r. Above it, the deterministic selection is reduced.
The SQL execution memory setting remains 4GB, as in the original pilot.

Outputs include run configuration/status, selected SQL identities and hashes,
factor population sizes, construction times, saved factors/sketches and
nearest-reference sensitivity reports. A failed query makes the run incomplete;
aggregate sensitivity is not silently calculated on a smaller sample.

Construction times exclude SQL execution, factor preparation and output writes.
Peak RSS is cumulative for the process, not independent memory per r.
This pilot evaluates Atomic membership KMV/CountSketch score and neighbour
sensitivity. It does not evaluate bags, PSD, calibrated detection metrics or
equivalence to a complete witness population. r=16 is not a full baseline.

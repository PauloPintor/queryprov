# Prepare provenance tokens

Run this after creating/importing the TPC-DS tables in DuckDB and before
generating or executing provenance queries. It adds a VARCHAR `prov` column
to each base table in the selected schema. Tokens are `prefix1` through
`prefixN`, where N is that table's row count. No separator is inserted.

Preview without changing the database:

```bash
python Generators/Database/add_provenance_tokens.py \
  --database /path/to/tpcds_prov.duckdb --schema tpcds
```

Apply to a backed-up database, with other writers stopped:

```bash
python Generators/Database/add_provenance_tokens.py \
  --database /path/to/tpcds_prov.duckdb --schema tpcds --apply
```

Validate afterwards using the same prefix configuration:

```bash
python Generators/Database/add_provenance_tokens.py \
  --database /path/to/tpcds_prov.duckdb --schema tpcds --check-only
```

The explicit prefix map is at the top of the script. It covers all 24 TPC-DS
tables. In particular `web_sales` uses `ws`, while `web_site` uses `web` to
avoid a collision. All prefixes contain only letters and are distinct; the
numeric suffix therefore also gives globally unique tokens across tables.
Unknown tables require an explicit prefix. Views are not modified.

Override any prefixes with a JSON object and pass `--prefixes-json` to all
commands. For example, `{"web_site": "wsite"}` changes only that prefix.
Repeated prefixes, null tokens, duplicates, missing suffixes, zero/negative
suffixes and values outside 1..N are rejected.

Existing valid `prov` columns are kept, not renumbered. Existing incompatible
columns cause failure; there is no overwrite option. Changes to all tables
are transactional: failure rolls back newly added columns and tokens.
Applying twice is safe if table contents and prefix settings have not changed.
Rows added later are not automatically annotated by this preparation script.

## Reproduction scope

This reconstructs the described rule, not the lost original implementation.
The default prefix map has not been verified against the historical database.
New tokens are assigned in DuckDB rowid order. This is repeatable within the
same unchanged database, but row-to-token assignments can differ after a fresh
import or database rebuild. Retain the annotated database to retain its exact
mapping. Never replace an experimental database's existing tokens or mix newly
numbered tokens with previously materialised provenance as though they matched.

This script does not download TPC-DS, build the source tables, alter the data
columns, execute workload SQLs, or regenerate Parquets/splits.

#!/usr/bin/env python3
"""Add table-prefixed sequential provenance tokens to an existing DuckDB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import duckdb


# Explicit and distinct; override with --prefixes-json if necessary.
PREFIXES = {
    'call_center': 'cc', 'catalog_page': 'cp', 'catalog_returns': 'cr',
    'catalog_sales': 'cs', 'customer': 'c', 'customer_address': 'ca',
    'customer_demographics': 'cd', 'date_dim': 'd',
    'household_demographics': 'hd', 'income_band': 'ib', 'inventory': 'inv',
    'item': 'i', 'promotion': 'p', 'reason': 'r', 'ship_mode': 'sm',
    'store': 's', 'store_returns': 'sr', 'store_sales': 'ss', 'time_dim': 't',
    'warehouse': 'w', 'web_page': 'wp', 'web_returns': 'wr',
    'web_sales': 'ws', 'web_site': 'web',
}


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def validate(connection, table, prefix, count):
    # N distinct positive suffixes bounded by N imply exactly 1..N.
    invalid = connection.execute(
        f'SELECT count(*) FROM {table} WHERE prov IS NULL '
        'OR NOT regexp_full_match(CAST(prov AS VARCHAR), ?) '
        'OR try_cast(substr(CAST(prov AS VARCHAR), ?) AS BIGINT) IS NULL '
        'OR try_cast(substr(CAST(prov AS VARCHAR), ?) AS BIGINT) > ?',
        [prefix + '[1-9][0-9]*', len(prefix) + 1, len(prefix) + 1, count],
    ).fetchone()[0]
    distinct = connection.execute(f'SELECT count(DISTINCT prov) FROM {table}').fetchone()[0]
    if invalid or distinct != count:
        raise ValueError(f'{table}: existing prov does not contain exactly {prefix}1..{prefix}{count}; nothing overwritten')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--schema', default='tpcds')
    parser.add_argument('--prefixes-json', type=Path, help='JSON object mapping table names to prefixes; overrides defaults')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true', help='Modify the database in one transaction; default is read-only preview')
    mode.add_argument('--check-only', action='store_true', help='Require and validate prov in every base table, without writing')
    args = parser.parse_args(argv)
    if not args.database.is_file():
        parser.error('An existing DuckDB database is required; this script does not create TPC-DS')
    prefixes = dict(PREFIXES)
    if args.prefixes_json:
        custom = json.loads(args.prefixes_json.read_text())
        if not isinstance(custom, dict):
            parser.error('Prefix file must be a JSON object')
        prefixes.update(custom)
    with duckdb.connect(str(args.database.resolve()), read_only=not args.apply) as connection:
        connection.execute('BEGIN TRANSACTION')
        try:
            tables = [row[0] for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog=current_database() "
                "AND table_schema=? AND table_type='BASE TABLE' ORDER BY table_name", [args.schema]).fetchall()]
            if not tables:
                raise ValueError(f'No base tables found in schema {args.schema!r}')
            used = {}
            plan = []
            for name in tables:
                prefix = prefixes.get(name)
                if not isinstance(prefix, str) or not re.fullmatch('[A-Za-z]+', prefix):
                    raise ValueError(f'{name}: supply an alphabetic prefix with --prefixes-json')
                if prefix in used:
                    raise ValueError(f'Prefix collision: {name} and {used[prefix]} use {prefix!r}')
                used[prefix] = name
                table = quote(args.schema) + '.' + quote(name)
                columns = [row[0].lower() for row in connection.execute(
                    'SELECT column_name FROM information_schema.columns '
                    'WHERE table_catalog=current_database() AND table_schema=? AND table_name=?',
                    [args.schema, name]).fetchall()]
                count = connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                exists = 'prov' in columns
                if exists:
                    validate(connection, table, prefix, count)
                elif args.check_only:
                    raise ValueError(f'{table}: missing prov column')
                elif 'rowid' in columns:
                    raise ValueError(f'{table}: an explicit rowid column shadows DuckDB row identifiers')
                plan.append((table, prefix, count, exists))
            # Check the whole plan before changing any table.
            for table, prefix, count, exists in plan:
                if exists:
                    print(f'[VALID, KEPT] {table}: {count:,} tokens')
                elif args.apply:
                    connection.execute(f'ALTER TABLE {table} ADD COLUMN prov VARCHAR')
                    connection.execute(
                        f'UPDATE {table} AS dst SET prov = ? || CAST(src.n AS VARCHAR) '
                        f'FROM (SELECT rowid AS rid, row_number() OVER (ORDER BY rowid) AS n FROM {table}) AS src '
                        'WHERE dst.rowid=src.rid', [prefix])
                    validate(connection, table, prefix, count)
                    print(f'[PREPARED] {table}: {count:,} tokens; prefix={prefix}')
                else:
                    print(f'[WOULD ADD] {table}: {count:,} tokens; prefix={prefix}')
            connection.execute('COMMIT')
        except BaseException:
            connection.execute('ROLLBACK')
            raise
    print('[DONE] ' + ('Transaction committed.' if args.apply else 'Read-only; database unchanged.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

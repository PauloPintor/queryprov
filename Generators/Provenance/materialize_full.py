#!/usr/bin/env python3
"""Materialise complete membership sets; never sample or truncate provenance.

Requires materialize_compact.py beside this script for the established identity
and token/witness encoding helpers. No compact summaries are constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import fcntl

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import materialize_compact as core


class ExpansionLimit(RuntimeError):
    pass


def terms(cell, layout):
    """Yield products of factors, each containing alternative token lists."""
    if cell is None:
        return
    if layout == 'product':
        yield [core.raw_factor_choices(part) for part in cell]
        return
    outer = cell if core.is_sequence(cell) else [cell]
    for term in outer:
        if term is None:
            continue
        nested = core.is_sequence(term) and any(core.is_sequence(x) for x in term)
        if nested:
            if layout != 'sum-products':
                raise ValueError('Nested provenance requires an explicit product layout; refusing to guess')
            yield [core.alternatives(part) for part in term]
        else:
            yield [[core.flat_tokens(term)]]


def collect(cursor, layout, representation, limit):
    column = core.provenance_column([d[0] for d in cursor.description])
    tokens = {}
    witnesses = {}
    candidates = 0
    nrows = 0
    while rows := cursor.fetchmany(64):
        nrows += len(rows)
        for row in rows:
            for factors in terms(row[column], layout):
                if not factors or any(not factor for factor in factors):
                    continue
                count = math.prod(map(len, factors))
                if representation != 'lineage':
                    candidates += count
                    if candidates > limit:
                        raise ExpansionLimit(f'Candidate witness combinations {candidates} exceed limit {limit}; no expansion/truncation of this product')
                encoded = []
                for factor in factors:
                    choices = []
                    for choice in factor:
                        ids = set()
                        for raw in choice:
                            token = raw.strip()
                            if not token:
                                continue
                            identifier = core.token_to_hash_i64(token)
                            previous = tokens.setdefault(identifier, token)
                            if previous != token:
                                raise ValueError('Token identifier collision')
                            ids.add(identifier)
                        choices.append(ids)
                    encoded.append(choices)
                if representation != 'lineage':
                    for selected in itertools.product(*encoded):
                        witness = tuple(sorted(set().union(*selected)))
                        if not witness:
                            continue
                        identifier = core.tuple_to_hash_i64(witness)
                        previous = witnesses.setdefault(identifier, witness)
                        if previous != witness:
                            raise ValueError('Witness identifier collision')
    result = {'nrows': nrows, 'representation_mode': 'full_membership',
              'provenance_layout': layout}
    if representation != 'atomic':
        result['lineage_hash'] = sorted(tokens)
    if representation != 'lineage':
        result['monoids_hash'] = sorted(witnesses)
    return result


def schema(representation):
    fields = [('query_id', pa.int64()), ('template_id', pa.int64()),
              ('query_name', pa.string()), ('subtype', pa.string()),
              ('nrows', pa.int64()), ('representation_mode', pa.string()),
              ('provenance_layout', pa.string())]
    if representation != 'atomic':
        fields.append(('lineage_hash', pa.list_(pa.int64())))
    if representation != 'lineage':
        fields.append(('monoids_hash', pa.list_(pa.int64())))
    return pa.schema(fields)


def worker(pipe, config):
    try:
        if config['address_space_gb']:
            import resource
            value = int(config['address_space_gb'] * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (value, value))
        connection = duckdb.connect(config['db'], read_only=True, config={
            'threads': config['threads'], 'memory_limit': config['memory_limit']})
        connection.execute('SET schema = ?', [config['schema']])
        pipe.send(('ready', None))
        while True:
            task = pipe.recv()
            if task is None:
                break
            sql, identity, layout = task
            try:
                result = collect(connection.execute(sql), layout, config['representation'],
                                 config['max_witness_combinations'])
                pipe.send(('ok', {**identity, **result}))
            except Exception as exc:
                pipe.send(('error', {'error_type': type(exc).__name__, 'error': str(exc)}))
        connection.close()
    except BaseException as exc:
        try:
            pipe.send(('error', {'error_type': type(exc).__name__, 'error': str(exc)}))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        pipe.close()


class Runner:
    def __init__(self, config):
        self.config = config
        self.process = None

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.pipe.close()
            self.process = None

    def receive(self):
        deadline = time.monotonic() + self.config['timeout_seconds']
        while time.monotonic() < deadline:
            if self.pipe.poll(0.1):
                try:
                    return self.pipe.recv()
                except EOFError:
                    break
            if not self.process.is_alive():
                break
        if self.process.is_alive():
            error = {'error_type': 'Timeout', 'error': 'Execution exceeded configured time limit'}
        else:
            error = {'error_type': 'WorkerExit', 'error': f'Worker exited: {self.process.exitcode}; inspect OS logs for cause'}
        self.close()
        return 'error', error

    def run(self, task):
        if self.process is None:
            context = mp.get_context('spawn')
            self.pipe, child = context.Pipe()
            self.process = context.Process(target=worker, args=(child, self.config))
            self.process.start()
            child.close()
            status, value = self.receive()
            if status != 'ready':
                self.close()
                return status, value
        self.pipe.send(task)
        status, value = self.receive()
        if status != 'ok':
            self.close()
        return status, value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--queries-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--queries', nargs='+', type=int, required=True)
    parser.add_argument('--pools', nargs='+', choices=['normal', 's1', 's2'], default=['normal', 's1', 's2'])
    parser.add_argument('--representation', choices=['both', 'lineage', 'atomic'], default='both')
    parser.add_argument('--max-witness-combinations', type=int, required=True,
                        help='Per-execution candidate expansion limit; not a sketch budget')
    parser.add_argument('--timeout-seconds', type=float, required=True)
    parser.add_argument('--memory-limit', default='8GB', help='DuckDB limit only')
    parser.add_argument('--address-space-gb', type=float, default=0,
                        help='Optional OS virtual-address-space cap per worker, 0 disables')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--schema', default='tpcds')
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--layout', choices=['auto', 'witnesses', 'product', 'sum-products'], default='auto',
                        help='auto: Q28 product, Q92/Q94 sum-products, other queries witnesses')
    args = parser.parse_args()
    if not args.schema.strip() or not math.isfinite(args.timeout_seconds) or not math.isfinite(args.address_space_gb):
        parser.error('Schema must be nonempty and limits must be finite')
    if not args.db.is_file() or not args.queries_root.is_dir():
        parser.error('Database and queries root must exist')
    if min(args.max_witness_combinations, args.timeout_seconds, args.threads, args.batch_size, *args.queries) <= 0 or args.address_space_gb < 0:
        parser.error('Limits, threads, batch size and query IDs must be positive')
    if len(set(args.queries)) != len(args.queries) or len(set(args.pools)) != len(args.pools):
        parser.error('Duplicate queries or pools')
    for key in ('db', 'queries_root', 'output_root'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output_root == args.queries_root or args.output_root in args.queries_root.parents or args.queries_root in args.output_root.parents:
        parser.error('Output must be a separate directory tree, outside the input tree')
    return args


def main():
    args = parse_args()
    config = vars(args).copy()
    config['db'] = str(args.db)
    stat = args.db.stat()
    fingerprint = core.digest_json({
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'db_size': stat.st_size, 'db_mtime_ns': stat.st_mtime_ns,
        'code': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'helpers': hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()})
    args.output_root.mkdir(parents=True, exist_ok=True)
    runner = Runner(config)
    failures = []
    try:
        with (args.output_root / '.full.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for query in args.queries:
                layout = args.layout if args.layout != 'auto' else ('product' if query == 28 else 'sum-products' if query in (92, 94) else 'witnesses')
                for pool in args.pools:
                    relative = Path(f'Q{query}') / f'q{query}_{core.POOL_SUFFIX[pool]}'
                    source, output = args.queries_root / relative, args.output_root / relative
                    paths = core.sql_files(source)
                    identities = core.resolve_identities(source, paths, query)
                    if len({row['query_id'] for row in identities.values()}) != len(identities):
                        raise ValueError(f'Duplicate execution IDs: {source}')
                    output.mkdir(parents=True, exist_ok=True)
                    context = {'fingerprint': fingerprint, 'sql': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, 'identities': identities}
                    manifest = output / '_full_run.json'
                    if manifest.exists():
                        if json.loads(manifest.read_text()) != context:
                            raise ValueError(f'Run/input mismatch: {output}; use a new output root')
                    elif any(output.iterdir()):
                        raise ValueError(f'Output is not empty: {output}')
                    core.write_json(manifest, context)
                    core.write_identity_manifest(output / 'compact_identities.csv', identities)
                    expected_batches = {f'batch_{i:04d}.parquet' for i in range(1, math.ceil(len(paths) / args.batch_size) + 1)}
                    if {p.name for p in output.glob('*.parquet')} - expected_batches:
                        raise ValueError(f'Unexpected Parquet files in {output}')
                    marker = output / '_SUCCESS.json'
                    if marker.exists():
                        marker.unlink()
                    pool_failed = False
                    for start in range(0, len(paths), args.batch_size):
                        selected = paths[start:start + args.batch_size]
                        target = output / f'batch_{start // args.batch_size + 1:04d}.parquet'
                        expected = [p.stem for p in selected]
                        if target.exists():
                            table = pq.read_table(target)
                            table.validate(full=True)
                            if table.schema.remove_metadata() != schema(args.representation) or table['query_name'].to_pylist() != expected or (table.schema.metadata or {}).get(b'full_run') != fingerprint.encode():
                                raise ValueError(f'Invalid existing batch: {target}')
                            continue
                        records = []
                        for path in selected:
                            status, value = runner.run((core.read_sql(path), identities[path.stem], layout))
                            if status != 'ok':
                                failures.append({'query': query, 'pool': pool, 'query_name': path.stem, **value})
                                core.write_json(args.output_root / 'full_errors.json', failures)
                                print(f'[FAILED] Q{query} {pool} {path.stem}: {value}', flush=True)
                                pool_failed = True
                                break
                            records.append(value)
                        if pool_failed:
                            # Do not publish a batch with missing executions or a partial population.
                            break
                        table = pa.Table.from_pylist(records, schema=schema(args.representation))
                        table = table.replace_schema_metadata({b'full_run': fingerprint.encode()})
                        temporary = target.with_suffix('.parquet.tmp')
                        pq.write_table(table, temporary, compression='zstd')
                        os.replace(temporary, target)
                        print(f'[WRITE] Q{query} {pool}: {start + len(selected)}/{len(paths)}', flush=True)
                    if not pool_failed:
                        core.write_json(marker, {'complete': True, 'executions': len(paths), 'fingerprint': fingerprint, 'representation': args.representation})
            core.write_json(args.output_root / 'full_status.json', {'complete': not failures, 'failures': failures})
    finally:
        runner.close()
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())

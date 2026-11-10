#!/usr/bin/env python3
"""Materialise complete Token TF and Witness TF, without PSD or sketches.

Keep materialize_full.py and materialize_compact.py in this directory.
"""
from materialize_full import main


if __name__ == '__main__':
    raise SystemExit(main(bags=True))

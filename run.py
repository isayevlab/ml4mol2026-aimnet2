#!/usr/bin/env python3
"""
Execute one jupytext source cell by cell, reporting where it fails.

    python run.py source/nb_04.py

Used to verify the material before a class. The three exercise notebooks stop
at their student blank with an AssertionError; that is the correct behaviour.
"""
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import jupytext

path = sys.argv[1]
nb = jupytext.read(path, fmt="py:percent")
g = {"__name__": "__main__"}
n = 0
t0 = time.perf_counter()
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    n += 1
    t = time.perf_counter()
    try:
        exec(compile(cell.source, f"{path}[{n}]", "exec"), g)
    except Exception:
        print(f"\n!!! cell {n} FAILED\n")
        print(cell.source[:600])
        print("-" * 50)
        traceback.print_exc()
        sys.exit(1)
    print(f"  [cell {n}] ok {time.perf_counter() - t:6.1f} s", flush=True)
print(f"{path}: {n} cells, {time.perf_counter() - t0:.0f} s")

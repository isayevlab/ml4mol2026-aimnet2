#!/usr/bin/env python3
"""
Execute one notebook cell by cell, reporting where it fails.

    python run.py source/nb_04.py
    python run.py solutions/3_activity_conformers_ANSWERS.ipynb

Accepts a jupytext percent-format source or an .ipynb file. Used to verify the
material before a class. The three exercise notebooks stop at their student
blank with an AssertionError; that is the correct behaviour.
"""
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import jupytext

path = sys.argv[1]
nb = jupytext.read(path)          # format is taken from the extension
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
    except KeyboardInterrupt:
        print(f"\n!!! interrupted in cell {n}")
        sys.exit(130)
    except SystemExit as exc:
        # a cell calling sys.exit() is a failure of the notebook, not a
        # clean end of this script
        print(f"\n!!! cell {n} called sys.exit({exc.code})\n")
        print(cell.source[:600])
        sys.exit(1)
    except Exception:
        print(f"\n!!! cell {n} FAILED\n")
        print(cell.source[:600])
        print("-" * 50)
        traceback.print_exc()
        sys.exit(1)
    print(f"  [cell {n}] ok {time.perf_counter() - t:6.1f} s", flush=True)
print(f"{path}: {n} cells, {time.perf_counter() - t0:.0f} s")

#!/usr/bin/env python3
"""
Build the notebooks and the answer keys from the jupytext sources.

    python build.py

Reads  source/nb_00.py .. source/nb_08.py   (jupytext percent format)
Writes notebooks/*.ipynb      the student notebooks
       solutions/*_ANSWERS.ipynb  the worked copies

An answer notebook is its exercise notebook with three changes: the answer-key
markdown from keys/<stem>.md prepended, the `SEAT` TODO comment replaced, and
the `ANSWER = ""` blank filled from ANSWERS below. Nothing else may differ, so
a fix to an exercise reaches its answer key automatically.
"""
import json
import pathlib
import re

import jupytext

ROOT = pathlib.Path(__file__).parent

NAMES = {
    "nb_00": "0_orientation",
    "nb_01": "1_energy",
    "nb_02": "2_geometry_optimisation",
    "nb_03": "3_activity_conformers",
    "nb_04": "4_molecular_dynamics",
    "nb_05": "5_reactions",
    "nb_06": "6_activity_ir_spectra",
    "nb_07": "7_activity_reactivity",
    "nb_08": "8_further_directions",
}

METADATA = {
    "jupytext": {"cell_metadata_filter": "-all", "main_language": "python",
                 "notebook_metadata_filter": "-all",
                 "text_representation": {"extension": ".py", "format_name": "percent"}},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

# The worked seat for each exercise, and the answer its student would write.
ANSWERS = {
    "3_activity_conformers": dict(
        seat="SEAT = 0          # butane, for this worked copy",
        answer='ANSWER = (\n'
        '    "Butane has two distinct conformers. The anti form, with the C-C-C-C torsion "\n'
        '    "at 180 degrees, is the global minimum and holds 64 percent of the population "\n'
        '    "at 298 K. The gauche form, at -64 degrees, lies 0.33 kcal/mol higher and "\n'
        '    "holds the remaining 36 percent. The ordering matches the textbook result, "\n'
        '    "although the computed gap is smaller than the experimental 0.6 to 0.9 "\n'
        '    "kcal/mol. Note also that the deduplication is invariant to reflection, so "\n'
        '    "the two mirror-image gauche wells are counted as one conformer and the 36 "\n'
        '    "percent is their combined population."\n'
        ')'),
    "6_activity_ir_spectra": dict(
        seat="SEAT = 0          # water, for this worked copy",
        answer='ANSWER = (\n'
        '    "The strongest band is the bend at 1597 cm-1, in which the two hydrogens move "\n'
        '    "and the oxygen barely does. It matches the experimental 1595 cm-1 almost "\n'
        '    "exactly. The two O-H stretches come out at 3791 and 3942 cm-1 against "\n'
        '    "experimental 3657 and 3756, an overestimate of about 4 percent that is "\n'
        '    "characteristic of the harmonic approximation for X-H bonds; scaling by 0.96 "\n'
        '    "brings them to 3639 and 3785. The pattern is general: the bend needs no "\n'
        '    "scaling and the stretches do."\n'
        ')'),
    "7_activity_reactivity": dict(
        seat="SEAT = 0          # methyl vinyl ketone, for this worked copy",
        answer='ANSWER = (\n'
        '    "The largest f+ falls on C4, the terminal CH2 carbon, at 0.232, well above "\n'
        '    "the carbonyl carbon C1 at 0.135. This is the carbon beta to the carbonyl and "\n'
        '    "exactly the site of conjugate addition predicted by the resonance "\n'
        '    "structures. The electrophilicity index is 1.06 eV. The carbonyl oxygen "\n'
        '    "carries the largest f- at 0.401, identifying it as the site a Lewis acid "\n'
        '    "would attack, which is also correct."\n'
        ')'),
}


def read_source(stem):
    nb = jupytext.read(ROOT / "source" / f"{stem}.py", fmt="py:percent")
    nb.metadata = dict(METADATA)
    # jupytext assigns a fresh random cell id on every read, which would dirty
    # every notebook on every build. Number the cells instead so that the
    # output depends on the source alone.
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"{stem}-{i:02d}"
    return nb


def build_notebooks():
    (ROOT / "notebooks").mkdir(exist_ok=True)
    for stem, name in NAMES.items():
        nb = read_source(stem)
        out = ROOT / "notebooks" / f"{name}.ipynb"
        # jupytext.write opens the file in text mode without newline="\n", so
        # on Windows it would emit CRLF; write the string ourselves to keep LF.
        text = jupytext.writes(nb, fmt="ipynb")
        out.write_text(text if text.endswith("\n") else text + "\n",
                       encoding="utf-8", newline="\n")
        print(f"  notebooks/{name}.ipynb")


def build_solutions():
    (ROOT / "solutions").mkdir(exist_ok=True)
    for name, spec in ANSWERS.items():
        nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
        key = (ROOT / "keys" / f"{name}.md").read_text(encoding="utf-8").rstrip("\n")

        replaced_seat = replaced_answer = False
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if src.startswith("SEAT = 0"):
                src = re.sub(r"^SEAT = 0.*$", spec["seat"], src, count=1, flags=re.M)
                replaced_seat = True
            if re.search(r'^ANSWER = ""', src, flags=re.M):
                src = re.sub(r'^ANSWER = "".*?(?=\n\nassert)', spec["answer"],
                             src, count=1, flags=re.M | re.S)
                replaced_answer = True
            cell["source"] = src.splitlines(keepends=True)

        assert replaced_seat, f"{name}: no SEAT line found"
        assert replaced_answer, f"{name}: no ANSWER blank found"
        assert "# TODO" not in json.dumps(nb), f"{name}: a TODO survived"

        nb["cells"].insert(0, {"cell_type": "markdown", "id": f"{name}-key",
                               "metadata": {},
                               "source": key.splitlines(keepends=True)})
        out = ROOT / "solutions" / f"{name}_ANSWERS.ipynb"
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8", newline="\n")
        print(f"  solutions/{name}_ANSWERS.ipynb")


print("notebooks:")
build_notebooks()
print("solutions:")
build_solutions()
print("\ndone. Run `python check_notebooks.py` next.")

# AIMNet2 hands-on tutorial

**AiMat Summer School 2026 — Machine Learning for Molecules**
Olexandr Isayev, Carnegie Mellon University

Ninety minutes, from a single-point energy to a predicted site of nucleophilic
attack. Nine notebooks that build on one another in one direction:

> energy → geometry optimisation → conformers → dynamics → reactions → infrared spectra → reactivity

Every concept is defined at the point it is first needed. The material assumes you
can read Python. It does not assume you have run a quantum chemistry calculation
before, and it does not assume a background in physical chemistry.

### 📖 **[Read the tutorial site](https://isayevlab.github.io/ml4mol2026-aimnet2/)**

The site has the install guide for macOS, Windows and Linux, the Colab route, the
troubleshooting page and the instructor run sheet. Start there. This README is the
short version.

## Quick start

**On Colab**, click a badge below. Runtime → Change runtime type → **T4 GPU** first.
The first cell installs everything and downloads the model parameters, about a
minute with no progress output. There is nothing to clone.

**Locally** (Python 3.11+; not Intel Macs, see the install guide):

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
git clone https://github.com/isayevlab/ml4mol2026-aimnet2 && cd ml4mol2026-aimnet2
jupyter lab
```

## The nine notebooks

| | Notebook | What it establishes | |
|---|---|---|---|
| | **0 Orientation** | the potential energy surface; what an MLIP is; the first calculation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isayevlab/ml4mol2026-aimnet2/blob/main/notebooks/0_orientation.ipynb) |
| Demo | **1 Energy, forces and charge** | force and partial charge defined; total charge is a network input, not a correction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isayevlab/ml4mol2026-aimnet2/blob/main/notebooks/1_energy.ipynb) |
| Demo | **2 Geometry optimisation** | local minimum, convergence; water to within 0.01 Å and 0.5°; the nearest minimum is not the best one | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isayevlab/ml4mol2026-aimnet2/blob/main/notebooks/2_geometry_optimisation.ipynb) |
| **Exercise 1** | **3 Conformational analysis** | conformer, ensemble, Boltzmann population. Eight systems, one per seat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isayevlab/ml4mol2026-aimnet2/blob/main/notebooks/3_activity_conformers.ipynb) |
| Demo | **4 Molecular dynamics** | timestep, thermostat, NVE, sampling error. Why a short trajectory resolves less than it seems to | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isayevlab/ml4mol2026-aimnet2/blob/main/notebooks/4_molecular_dynamics.ipynb) |
| Demo | **5 Reactions and transition states** | saddle point, barrier, Hessian, imaginary frequency. An S<sub>N</sub>2 reaction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isayevlab/ml4mol2026-aimnet2/blob/main/notebooks/5_reactions.ipynb) |
| **Exercise 2** | **6 Infrared spectra** | normal mode, harmonic approximation, selection rule. Eight molecules, one per seat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isayevlab/ml4mol2026-aimnet2/blob/main/notebooks/6_activity_ir_spectra.ipynb) |
| **Exercise 3** | **7 Chemical reactivity** | *I*, *A*, *μ*, *η*, electrophilicity index, Fukui function. Eight molecules, one per seat | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isayevlab/ml4mol2026-aimnet2/blob/main/notebooks/7_activity_reactivity.ipynb) |
| | **8 Further directions** | periodic systems, bond dissociation energies, Pd, OpenMM, pysisyphus | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isayevlab/ml4mol2026-aimnet2/blob/main/notebooks/8_further_directions.ipynb) |

Each notebook installs its own dependencies in the first cell and stands alone, so a
student who arrives late or whose kernel dies can open any notebook and start there.

## The three exercises

Each runs about 13 minutes and hands out eight systems by seat number, so no two
neighbours compute the same thing and the room pools its results into one table.
`solutions/` holds a worked notebook for each, with the expected result for all
eight seats, the points to draw out, and the difficulties students actually hit.

| | Question the student answers | The eight systems |
|---|---|---|
| **1** | Which conformer is populated at room temperature, and by how much? | butane, methylcyclohexane, ethylene glycol, glycine, 1,2-dichloroethane, 1,4-butanediol, ibuprofen, alanine dipeptide |
| **2** | Which vibration gives the strongest infrared band, and where is it? | water, formaldehyde, methanol, acetone, acetonitrile, chloroform, benzene, ethanol |
| **3** | Where does a nucleophile attack, and how electrophilic is the molecule? | methyl vinyl ketone, acrolein, acrylonitrile, acetone, phenol, aniline, anisole, nitrobenzene |

## Slides

`slides/index.html` is a single self-contained file, 33 slides, Carnegie Mellon
palette and typefaces on white. Arrow keys or click to advance, <kbd>O</kbd> for an
overview grid, <kbd>F</kbd> for fullscreen, <kbd>P</kbd> to print to PDF, `#/12` to
deep-link a slide. It works offline from a local file, so a dead conference network
is not a problem.

## Layout

```
index.html  install.html  notebooks.html      the site, served by GitHub Pages
troubleshooting.html  teaching.html
slides/index.html      33 slides, self-contained
notebooks/             the nine tutorial notebooks
solutions/             worked answers for the three exercises
keys/                  the answer-key text, prepended to the solutions at build time
source/                jupytext .py sources the notebooks are generated from
site/                  the site generator
build.py               sources -> notebooks/ and solutions/
check_notebooks.py     static check, run before class
predeploy.py           link and style check, run before pushing
run.py                 execute one source file cell by cell
```

`source/` and `site/` are optional. The notebooks are the deliverable; the `.py`
files are there if you would rather edit percent-format scripts than JSON.

```bash
python build.py            # regenerate notebooks/ and solutions/ from source/
python check_notebooks.py  # static check of the notebooks and the site
python predeploy.py        # link check and house-style check, run before pushing
python run.py source/nb_05.py   # execute one notebook cell by cell
```

An answer notebook is its exercise notebook with the answer-key markdown from
`keys/` prepended and its two blanks filled. Nothing else may differ, so a
correction to an exercise reaches its answer key automatically.

## Publishing

Settings → Pages → Deploy from branch → `main`, folder `/`. The site is then at
`https://<user>.github.io/<repo>/` and the deck at `.../slides/`.

## Credits

Olexandr Isayev, Carnegie Mellon University.
AiMat Summer School 2026, Machine Learning for Molecules.
Materials MIT licensed.

AIMNet2: Anstine, Zubatyuk and Isayev, *Chem. Sci.* **2025**, *16*, 10228.
AIMNet2-NSE: Kalita et al., *Angew. Chem. Int. Ed.* **2026**, e202516763.
Package and models: https://github.com/isayevlab/aimnetcentral

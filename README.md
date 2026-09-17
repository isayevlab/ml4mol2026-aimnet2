# AIMNet2 hands-on tutorial

**AiMat Summer School 2026 · Machine Learning for Molecules**
Olexandr Isayev, Carnegie Mellon University

[![Tutorial site](https://img.shields.io/badge/site-isayevlab.github.io-C41230)](https://isayevlab.github.io/ml4mol2026-aimnet2/)
[![Slides](https://img.shields.io/badge/slides-33-6D6E71)](https://isayevlab.github.io/ml4mol2026-aimnet2/slides/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isayevlab/ml4mol2026-aimnet2/blob/main/notebooks/0_orientation.ipynb)
[![check](https://github.com/isayevlab/ml4mol2026-aimnet2/actions/workflows/check.yml/badge.svg)](https://github.com/isayevlab/ml4mol2026-aimnet2/actions/workflows/check.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Ninety minutes, from a single-point energy to a predicted site of nucleophilic
attack, with one machine-learned interatomic potential. Nine notebooks that build
on one another in one direction:

> energy → geometry optimisation → conformers → dynamics → reactions → infrared spectra → reactivity

Every concept is defined at the point it is first needed. The material assumes
you can read Python. It does not assume you have run a quantum chemistry
calculation before, or a background in physical chemistry.

**Start at the [tutorial site](https://isayevlab.github.io/ml4mol2026-aimnet2/).**
It has the install guide for macOS, Windows and Linux, the Colab route, a
troubleshooting page and the instructor run sheet. This README is the short version.

## Contents

- [Quick start](#quick-start)
- [The nine notebooks](#the-nine-notebooks)
- [The three exercises](#the-three-exercises)
- [Slides](#slides)
- [Repository layout](#repository-layout)
- [Editing and building](#editing-and-building)
- [Citing](#citing)

## Quick start

### On Google Colab, nothing to install

Click a badge in the table below. Then **Runtime → Change runtime type → T4 GPU**,
and restart the runtime if it had already started. The first cell installs
everything and downloads the model parameters, about a minute with no progress
output. There is nothing to clone.

### Locally

Python 3.11 or later on Linux, Windows or Apple Silicon macOS. Intel Macs are not
supported: PyTorch no longer ships wheels for them.

```bash
git clone https://github.com/isayevlab/ml4mol2026-aimnet2
cd ml4mol2026-aimnet2
python3 -m venv .venv && source .venv/bin/activate     # Windows: py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Two things the install guide covers in more detail:

- **Linux with an NVIDIA GPU.** The default PyTorch wheel is built for CUDA 13
  and needs driver 580 or newer. On an older driver, run
  `pip install torch --index-url https://download.pytorch.org/whl/cu126` first.
  Without an NVIDIA GPU, `--index-url https://download.pytorch.org/whl/cpu` cuts
  the download from about 3 GB to about 200 MB.
- **Windows.** The PyPI wheel is CPU-only, and `torch.compile` is unavailable.
  The setup cell in every notebook disables it automatically. Everything runs;
  it is five to twenty times slower than a GPU.

Every notebook installs its own dependencies in its first cell and stands alone,
so a student who arrives late, or whose kernel dies, can open any notebook and
start there.

Wherever the structure itself is the point, the notebook draws it in an
interactive 3D viewer (py3Dmol): the first molecule, charges on atoms, an
optimisation as a film, the lowest conformers, an MD trajectory, a transition
state moving along its imaginary mode, the strongest infrared mode, Fukui indices,
a periodic water box. The viewer needs a network connection; every result is also
printed as text.

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

[`slides/index.html`](slides/index.html) is a single file: 33 slides, Carnegie
Mellon palette on white. Arrow keys or click to advance, <kbd>O</kbd> for an
overview grid, <kbd>F</kbd> for fullscreen, <kbd>P</kbd> to print to PDF, `#/12`
to deep-link a slide. It renders from a local file with no network; only the
typefaces fall back to system fonts.

## Repository layout

```
notebooks/             the nine tutorial notebooks (the deliverable)
solutions/             worked answers for the three exercises
keys/                  answer-key text, prepended to the solutions at build time
source/                jupytext percent-format sources the notebooks are built from
slides/index.html      the deck
index.html install.html notebooks.html troubleshooting.html teaching.html
                       the site, served by GitHub Pages from the repository root
site/                  the site generator
build.py               source/ + keys/  ->  notebooks/ and solutions/
check_notebooks.py     static check of the notebooks, run before class
predeploy.py           link and house-style check, run before pushing
run.py                 execute one notebook or source cell by cell
```

## Editing and building

The notebooks are generated. Edit the `.py` files in `source/`, then:

```bash
python build.py            # regenerate notebooks/ and solutions/
python check_notebooks.py  # static check of the notebooks
python predeploy.py        # link check and house-style check
python run.py source/nb_05.py                              # execute one source, cell by cell
python run.py solutions/6_activity_ir_spectra_ANSWERS.ipynb  # or one answer notebook
```

Builds are deterministic, and the CI workflow fails if `notebooks/` and
`solutions/` are out of step with `source/`. An answer notebook is its exercise
notebook with the key from `keys/` prepended and its two blanks filled. Nothing
else may differ, so a correction to an exercise reaches its answer key
automatically.

The site is generated by `python site/build_site.py`. GitHub Pages serves the
repository root of `main`.

## Citing

If this material is useful in your own teaching, cite the repository
([`CITATION.cff`](CITATION.cff)) and the AIMNet2 papers:

- Anstine, D. M.; Zubatyuk, R.; Isayev, O. AIMNet2: a neural network potential
  to meet your neutral, charged, organic, and elemental-organic needs.
  *Chem. Sci.* **2025**, *16*, 10228. [10.1039/D4SC08572H](https://doi.org/10.1039/D4SC08572H)
- Kalita, B. *et al.* AIMNet2-NSE. *Angew. Chem. Int. Ed.* **2026**, *65*, e202516763.
  [10.1002/anie.202516763](https://doi.org/10.1002/anie.202516763)

Package and models: [isayevlab/aimnetcentral](https://github.com/isayevlab/aimnetcentral).

## Credits

Olexandr Isayev, Carnegie Mellon University. AiMat Summer School 2026, Machine
Learning for Molecules. Materials are MIT licensed.

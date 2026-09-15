# %% [markdown]
# # 8. Further directions
#
# Everything in this tutorial has used an isolated molecule in the gas phase,
# evaluated through ASE. This notebook sketches what lies immediately beyond,
# with working code where it is short enough to be useful.

# %% [markdown]
# ## Before you begin
#
# Run the cell below. On a fresh machine it takes roughly one minute: it
# installs the `aimnet` package, downloads the trained model parameters, and on
# a GPU compiles a CUDA kernel the first time one is required. These steps occur
# once per machine and give no progress output, so the cell will appear to do
# nothing for some time. This is expected.

# %%
import subprocess, sys, warnings
warnings.filterwarnings("ignore")

def _pip(*packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)

try:
    import aimnet, rdkit, ase          # noqa: F401
except ImportError:
    _pip("aimnet[ase]", "rdkit")
import numpy as np
import torch
from aimnet.calculators import AIMNet2Calculator, AIMNet2ASE

GPU = torch.cuda.is_available()
assert sys.version_info >= (3, 11), "aimnet requires Python 3.11 or later"
print(f"Python {sys.version.split()[0]}   PyTorch {torch.__version__}   GPU available: {GPU}")

for _name in ("aimnet2-2025", "aimnet2-nse"):   # the two models used below
    _ = AIMNet2Calculator(_name)
print("Models loaded.")
# %% [markdown]
# ### Two helper functions
#
# These appear in every notebook of this tutorial. They are short deliberately:
# nothing in this material is hidden from you.

# %%
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

def build(smiles, charge=None, mult=1, seed=42):
    """
    Construct a three-dimensional structure from a SMILES string.

    SMILES is a line notation for chemical structure: `CCO` denotes ethanol.
    It encodes which atoms are bonded to which, but not where they are in
    space. RDKit generates a plausible set of coordinates using distance
    geometry, then relaxes it with the MMFF force field. The result is a
    starting point, not a converged structure.

    Returns an ASE `Atoms` object. ASE (Atomic Simulation Environment) is the
    standard Python container for a molecular structure and the calculator
    attached to it.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3(); params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    atoms = Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
                  positions=mol.GetConformer().GetPositions())
    # AIMNet2 reads the total charge and spin multiplicity from atoms.info
    # on every evaluation, so they are stored there rather than passed around.
    atoms.info["charge"] = Chem.GetFormalCharge(mol) if charge is None else int(charge)
    atoms.info["mult"] = int(mult)
    return atoms

def attach(atoms, model="aimnet2"):
    """Attach an AIMNet2 calculator to a structure, so that energies and
    forces can be requested from it through the standard ASE interface."""
    atoms.calc = AIMNet2ASE(AIMNet2Calculator(model),
                            charge=atoms.info.get("charge", 0),
                            mult=atoms.info.get("mult", 1))
    return atoms

EV2KCAL = 23.060548     # kcal per mol, per eV
KT_298 = 0.5924         # kT at 298.15 K, in kcal/mol
FMAX = 0.02             # force convergence threshold, eV per Angstrom

# %% [markdown]
# ## 8.1 Periodic systems
#
# > **Definition — periodic boundary conditions.**
# > A simulation cell is repeated infinitely in all directions, so that a finite
# > number of atoms represents a bulk phase. This is how liquids and crystals
# > are modelled.
#
# Long-range electrostatics require care under periodic boundary conditions: a
# charge distribution in a repeating cell needs a lattice sum rather than a
# distance cutoff. AIMNet2 implements four (simple, DSF, Ewald and PME); two are compared below.

# %%
from ase.build import molecule as ase_molecule

box = 9.0
rng = np.random.default_rng(0)
cell = None
for _ in range(6):
    w = ase_molecule("H2O")
    w.translate(rng.uniform(1.5, box - 1.5, 3))
    cell = w if cell is None else cell + w
cell.set_cell([box, box, box]); cell.set_pbc(True)
cell.info.update(charge=0, mult=1)

base = AIMNet2Calculator("aimnet2-2025")
for method in ("dsf", "ewald"):
    base.set_lrcoulomb_method(method, cutoff=box / 2)
    cell.calc = AIMNet2ASE(base, charge=0, mult=1)
    print(f"{method:<8}E = {cell.get_potential_energy():14.4f} eV")

print("\nDSF is a damped, shifted real-space sum: inexpensive and approximate.")
print("Ewald evaluates the lattice sum properly: more costly and more accurate.")

# %% [markdown]
# For a molecular crystal, read a structure from a CIF file with `ase.io.read`,
# attach the calculator, and relax with `ase.filters.FrechetCellFilter` so that
# the cell parameters optimise alongside the atomic positions.
#
# ## 8.2 Open-shell chemistry: bond dissociation energies
#
# > **Definition — bond dissociation energy (BDE).**
# > The energy required to break a bond homolytically, giving two radicals:
# > R–H → R• + H•. It is a direct measure of bond strength.
#
# Radicals have an unpaired electron and therefore a multiplicity of 2. The
# `aimnet2-nse` model handles them.

# %%
from ase.optimize import LBFGS

def optimised_energy(smiles, charge=0, mult=1):
    a = build(smiles, charge=charge, mult=mult)
    a.calc = AIMNet2ASE(AIMNet2Calculator("aimnet2-nse"), charge=charge, mult=mult)
    LBFGS(a, logfile=None).run(fmax=FMAX, steps=600)
    return a.get_potential_energy()

hydrogen = Atoms("H", positions=[[0, 0, 0]])
hydrogen.info.update(charge=0, mult=2)
hydrogen.calc = AIMNet2ASE(AIMNet2Calculator("aimnet2-nse"), charge=0, mult=2)
E_H = hydrogen.get_potential_energy()

print(f"{'bond':<32}{'computed':>12}{'literature':>13}")
print("-" * 57)
for label, parent, radical, literature in (
        ("CH3-H   in methane",           "C",           "[CH3]",         105),
        ("PhCH2-H in toluene, benzylic", "Cc1ccccc1",   "[CH2]c1ccccc1",  90),
        ("HO-H    in water",             "O",           "[OH]",          119)):
    bde = (optimised_energy(radical, mult=2) + E_H - optimised_energy(parent)) * EV2KCAL
    print(f"{label:<32}{bde:>12.1f}{literature:>13}")

print("\nAll in kcal/mol. The ordering is the chemically important part: a benzylic")
print("C-H bond is much weaker than one in methane, because the resulting radical is")
print("delocalised over the aromatic ring. That difference is the basis of selectivity")
print("in radical hydrogen-abstraction chemistry.")

# %% [markdown]
# ## 8.3 Other directions, without code
#
# **Palladium catalysis.** The `aimnet2-pd` model is the only member of the
# family containing a transition metal. It is also the only one that is not a
# gas-phase model: implicit THF solvation is present in its training data. Its
# energies are therefore not comparable with those of any other family member.
#
# **Long-timescale dynamics.** ASE is adequate for picoseconds. For nanoseconds
# of a solvated system, drive the model from a specialised engine:
#
# ```python
# from openmmml import MLPotential
# potential = MLPotential("aimnet2")
# system = potential.createSystem(topology)
# ```
#
# **Reaction paths at scale.** `pip install "aimnet[pysis]"` makes nudged
# elastic band, growing string, transition-state optimisation and intrinsic
# reaction coordinate calculations available through pysisyphus configuration
# files.
#
# **Large numbers of molecules.** `Auto3D` already depends on AIMNet2 and uses
# it by default: SMILES in, optimised and ranked three-dimensional structures
# out, processed in batches.
#
# **Training on your own data.** `aimnet train --config your.yaml` exists and is
# documented. The published models were trained on tens of millions of reference
# calculations; fine-tuning on a specific system is a realistic project, whereas
# training from scratch is not a short one.
#
# ## Where to continue
#
# - repository: `github.com/isayevlab/aimnetcentral`
# - documentation: `isayevlab.github.io/aimnetcentral`
# - browser demonstration, no installation: `huggingface.co/spaces/isayevlab/aimnet2-demo`
#
# The package on PyPI is **`aimnet`**. The repository `isayevlab/AIMNet2` is
# deprecated, and any instructions that begin by cloning it are out of date.
#
# ## Primary references
#
# - Anstine, D. M.; Zubatyuk, R.; Isayev, O. *Chem. Sci.* **2025**, *16*, 10228.
# - Kalita, B. *et al.* *Angew. Chem. Int. Ed.* **2026**, e202516763.
# - Parr, R. G.; Szentpály, L. v.; Liu, S. *J. Am. Chem. Soc.* **1999**, *121*, 1922
#   (the electrophilicity index).
# - Parr, R. G.; Yang, W. *J. Am. Chem. Soc.* **1984**, *106*, 4049 (Fukui functions).

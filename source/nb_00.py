# %% [markdown]
# # 0. Orientation
#
# **AIMNet2 hands-on tutorial** · AiMat Summer School 2026, *Machine Learning
# for Molecules*
#
# This tutorial assumes you can read Python. It does not assume you have run a
# quantum chemistry calculation before, and it does not assume a background in
# physical chemistry. Every concept is defined where it is first needed.
#
# ---
#
# ## What this tutorial is about
#
# Almost everything one wants to know about a molecule follows from a single
# function: the **potential energy** of the molecule as a function of where its
# atoms are.
#
# > **Definition — potential energy surface (PES).**
# > For a molecule of *N* atoms, the potential energy *E* depends on the 3*N*
# > Cartesian coordinates of those atoms. The function *E*(**R**) is called the
# > potential energy surface. Every structural and thermodynamic property in
# > this tutorial is a statement about the shape of that surface.
#
# From the shape of the PES one obtains:
#
# | Feature of the surface | Chemical meaning |
# |---|---|
# | a local minimum | a stable structure: a conformer or an isomer |
# | the depth of one minimum relative to another | which form is populated, and in what ratio |
# | the curvature at a minimum | the vibrational frequencies, hence the infrared spectrum |
# | a first-order saddle point between two minima | a transition state, and an activation barrier |
# | motion on the surface at finite temperature | molecular dynamics |
#
# The surface itself comes from quantum mechanics. Solving the electronic
# Schrödinger equation accurately for every geometry is far too slow for any of
# the tasks above. The practical compromise for the last fifty years has been
# density functional theory (DFT), which is accurate but still costs seconds to
# hours per structure.
#
# > **Definition — machine-learned interatomic potential (MLIP).**
# > A model, here a neural network, trained to reproduce the energies and forces
# > of a reference quantum chemistry method. Once trained it evaluates in
# > milliseconds rather than minutes, at close to the accuracy of the method it
# > was trained on.
#
# **AIMNet2** is such a model. It was trained on approximately twenty million
# DFT calculations and covers fourteen elements: H, B, C, N, O, F, Si, P, S, Cl,
# As, Se, Br and I.
#
# ## How this tutorial progresses
#
# Each notebook uses only what the previous ones established.
#
# | | Notebook | Introduces |
# |---|---|---|
# | 1 | Energy | forces, partial charges, charge and spin as inputs |
# | 2 | Geometry optimisation | local minima, convergence |
# | **3** | **Conformational analysis** *(your turn)* | conformers, Boltzmann populations |
# | 4 | Molecular dynamics | temperature, trajectories, sampling |
# | 5 | Reactions | transition states, activation barriers, the Hessian |
# | **6** | **Infrared spectra** *(your turn)* | normal modes, the harmonic approximation, selection rules |
# | **7** | **Chemical reactivity** *(your turn)* | ionisation potential, electron affinity, Fukui functions |
# | 8 | Further directions | periodic systems, long-timescale dynamics, larger workflows |
#
# ---

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

_ = AIMNet2Calculator("aimnet2")          # downloads parameters on first use
print("Model loaded.")
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
# ## A first calculation
#
# Ethanol. Three inputs: the positions of the atoms, their atomic numbers, and
# the total charge of the molecule.

# %%
ethanol = build("CCO")
attach(ethanol)

energy = ethanol.get_potential_energy()
print(f"number of atoms   {len(ethanol)}")
print(f"total charge      {ethanol.info['charge']}")
print(f"potential energy  {energy:.4f} eV")

# %% [markdown]
# ### On units, and on what that number means
#
# Energies here are in **electronvolts (eV)**; distances in **ångström (Å)**,
# where 1 Å = 10⁻¹⁰ m, roughly the length of a chemical bond.
#
# Chemists more often use kcal/mol: 1 eV = 23.06 kcal/mol. A useful reference
# scale: a hydrogen bond is worth about 5 kcal/mol, a carbon–carbon single bond
# about 85 kcal/mol, and thermal energy at room temperature is 0.59 kcal/mol.
#
# The number printed above is large and negative because it is a **total**
# electronic energy, measured relative to infinitely separated nuclei and
# electrons. Its
# absolute value carries no useful information. **Only differences between
# energies of the same set of atoms are meaningful**, and every quantity
# computed in this tutorial is such a difference.
#
# ## Verification
#
# Run this before continuing. If any line reports a failure, ask for help now
# rather than in twenty minutes.

# %%
charges = ethanol.get_charges()
forces = ethanol.get_forces()

checks = [
    ("aimnet imports and evaluates", np.isfinite(energy) and abs(energy) > 1),
    ("forces have shape (N, 3)",     forces.shape == (len(ethanol), 3)),
    ("partial charges sum to the total charge",
                                     abs(charges.sum() - ethanol.info["charge"]) < 1e-3),
    ("GPU available",                GPU),
]
for label, passed in checks:
    print(f"{'pass  ' if passed else '----  '}{label}")

if all(p for _, p in checks):
    print("\nReady.")
elif all(p for l, p in checks if l != "GPU available"):
    print("\nReady. No GPU is available, so calculations will run on the CPU.")
    print("Everything in this tutorial still works; it is roughly five to twenty times slower.")
else:
    print("\nSomething is wrong. Please ask for help.")

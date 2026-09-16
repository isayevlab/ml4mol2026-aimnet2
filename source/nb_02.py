# %% [markdown]
# # 2. Geometry optimisation
#
# *Demonstration notebook.*
#
# ## Concepts introduced here
#
# > **Definition — local minimum.**
# > A point on the potential energy surface at which all forces vanish and the
# > energy increases in every direction. Physically, a stable structure: the
# > molecule placed there will stay there in the absence of thermal motion.
#
# > **Definition — geometry optimisation.**
# > The iterative procedure of moving atoms along the forces until the forces
# > are (numerically) zero. Each step requires one energy and force evaluation.
# > Because the forces point downhill, the procedure finds a *local* minimum:
# > the one nearest the starting structure, not necessarily the lowest one.
#
# > **Definition — convergence criterion.**
# > Optimisation stops when the largest force on any atom falls below a
# > threshold, conventionally written `fmax`. Here `fmax = 0.02 eV/Å`
# > throughout. AIMNet2 evaluates in single precision, so forces below roughly
# > 0.01 eV/Å are comparable to the model's own numerical noise; requiring a
# > tighter threshold costs several times more steps and changes no chemical
# > conclusion.
#
# The starting structures used here come from RDKit distance geometry followed
# by an MMFF force-field relaxation. MMFF is a classical force field: fast, but
# with no electronic structure behind it. Its geometries are a reasonable
# starting guess and nothing more.

# %% [markdown]
# ## Before you begin
#
# Run the cell below. On a fresh machine it takes roughly one minute: it
# installs the `aimnet` package, downloads the trained model parameters, and on
# a GPU compiles a CUDA kernel the first time one is required. These steps occur
# once per machine and give no progress output, so the cell will appear to do
# nothing for some time. This is expected.

# %%
import os, subprocess, sys, warnings
warnings.filterwarnings("ignore")
assert sys.version_info >= (3, 11), "aimnet requires Python 3.11 or later"

if sys.platform == "win32":
    # torch.compile needs a C++ compiler, which Windows machines rarely have.
    # Disabling it must happen before torch is imported.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

def _pip(*packages):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:]); print(r.stderr[-4000:])
        raise RuntimeError("pip install failed; see the output above")

try:
    import aimnet, rdkit, ase          # noqa: F401
except ImportError:
    _pip("aimnet[ase]", "rdkit", "warp-lang<1.18")
import numpy as np
import torch
from aimnet.calculators import AIMNet2Calculator, AIMNet2ASE

GPU = torch.cuda.is_available()
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

_MODELS = {}

def attach(atoms, model="aimnet2"):
    """Attach an AIMNet2 calculator to a structure, so that energies and
    forces can be requested from it through the standard ASE interface.
    Loading a model is the slow step, so each model is loaded once and
    shared by every structure that uses it."""
    if model not in _MODELS:
        _MODELS[model] = AIMNet2Calculator(model)
    atoms.calc = AIMNet2ASE(_MODELS[model],
                            charge=atoms.info.get("charge", 0),
                            mult=atoms.info.get("mult", 1))
    return atoms

EV2KCAL = 23.060548     # kcal per mol, per eV
KT_298 = 0.5924         # kT at 298.15 K, in kcal/mol
FMAX = 0.02             # force convergence threshold, eV per Angstrom

# %% [markdown]
# ## 2.1 One optimisation, step by step
#
# Water. The starting structure comes from the force field; the optimisation
# refines it against AIMNet2.

# %%
from ase.optimize import LBFGS

water = attach(build("O"))
print(f"before:  O-H {water.get_distance(0,1):.4f} and {water.get_distance(0,2):.4f} A,"
      f"  H-O-H {water.get_angle(1,0,2):.2f} deg")
print(f"         energy {water.get_potential_energy():.4f} eV,"
      f"  largest force {np.abs(water.get_forces()).max():.4f} eV/A\n")

opt = LBFGS(water, logfile="-")            # logfile="-" prints each step
opt.run(fmax=FMAX, steps=100)

print(f"\nafter:   O-H {water.get_distance(0,1):.4f} and {water.get_distance(0,2):.4f} A,"
      f"  H-O-H {water.get_angle(1,0,2):.2f} deg")
print(f"         energy {water.get_potential_energy():.4f} eV,"
      f"  largest force {np.abs(water.get_forces()).max():.4f} eV/A")
print("\nexperimental gas-phase water: O-H 0.958 A, H-O-H 104.5 deg")

# %% [markdown]
# Each line of the log is one step: the optimiser evaluates the energy and
# forces, moves the atoms downhill, and repeats. The energy decreases
# monotonically and the largest force falls below the threshold.
#
# The converged bond length and angle should be within about 0.01 Å and 1° of
# the experimental gas-phase values.
#
# ## 2.2 Cost
#
# The reason to use a machine-learned potential rather than DFT directly is
# that an optimisation needs tens to hundreds of energy-and-force evaluations.

# %%
import time

for smiles, label in (("CCO", "ethanol"),
                      ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "ibuprofen"),
                      ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "caffeine")):
    a = attach(build(smiles))
    t0 = time.perf_counter()
    o = LBFGS(a, logfile=None); o.run(fmax=FMAX, steps=500)
    dt = time.perf_counter() - t0
    print(f"{label:<12}{len(a):>4} atoms{o.get_number_of_steps():>6} steps"
          f"{dt:>8.1f} s{dt/max(o.get_number_of_steps(),1)*1000:>9.0f} ms/step")

# %% [markdown]
# The same optimisations with a hybrid density functional would take minutes to
# hours each. This difference in cost is what makes the rest of this tutorial
# possible inside ninety minutes.
#
# ## 2.3 An optimisation finds the nearest minimum, not the best one
#
# This is the single most important practical caveat about geometry
# optimisation, and it is the reason notebook 3 exists.
#
# Butane has a rotatable central C–C bond. Started from different torsion
# angles, the same optimiser reaches different structures, each a genuine
# minimum.
#
# > **Definition — torsion angle (dihedral).**
# > For four bonded atoms A–B–C–D, the angle between the plane containing
# > A, B, C and the plane containing B, C, D. It describes rotation about the
# > central B–C bond.

# %%
from rdkit.Chem import rdMolTransforms

def butane_at(start_angle):
    """Build butane with the central C-C-C-C torsion set to a chosen value.

    The torsion must be set on the RDKit molecule, not on the ASE structure.
    RDKit rotates the whole substituent about the bond; ASE's set_dihedral moves
    a single atom unless told otherwise, which would simply break the molecule.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    AllChem.EmbedMolecule(mol, p); AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, 0, 1, 2, 3, float(start_angle))
    atoms = Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
                  positions=conf.GetPositions())
    atoms.info.update(charge=0, mult=1)
    return attach(atoms)

def signed(angle):
    return angle - 360 if angle > 180 else angle

print(f"{'start':>8}{'final':>10}{'steps':>8}{'relative energy':>18}")
print("-" * 44)
results = []
for start in (0, 60, 90, 180):
    a = butane_at(start)
    o = LBFGS(a, logfile=None); o.run(fmax=FMAX, steps=400)
    results.append((start, signed(a.get_dihedral(0, 1, 2, 3)),
                    a.get_potential_energy(), o.get_number_of_steps()))

lowest = min(r[2] for r in results)
for start, final, e, steps in results:
    print(f"{start:>4} deg{final:>6.0f} deg{steps:>8}"
          f"{(e - lowest) * EV2KCAL:>9.2f} kcal/mol")

# %% [markdown]
# Two distinct minima appear, near 180° and near ±60°.
#
# > **Definitions — anti and gauche.**
# > For a four-atom chain, the conformation with a torsion angle near 180° is
# > called *anti*; those near ±60° are called *gauche*. In butane the anti form
# > is lower in energy because the two methyl groups are as far apart as
# > possible.
#
# The optimiser reached whichever minimum was closest to where it started. It
# had no way to find the others, because moving between them requires going
# *uphill* first, and the forces always point downhill.
#
# **Therefore: optimising one structure tells you about one minimum.** To say
# anything about a flexible molecule you must find and compare many. That is
# the subject of the next notebook, and it is your first exercise.

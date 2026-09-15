# %% [markdown]
# # 1. Energy, forces and charge
#
# *Demonstration notebook.*
#
# ## Concepts introduced here
#
# > **Definition — force.**
# > The force on atom *i* is the negative gradient of the potential energy with
# > respect to that atom's position:
# >
# > **F**ᵢ = −∂*E*/∂**R**ᵢ
# >
# > It points in the direction in which that atom would move to lower the
# > energy. Forces are reported in eV/Å. A structure at which every force is
# > zero is a **stationary point** on the potential energy surface.
#
# > **Definition — partial atomic charge.**
# > Electrons are not owned by individual atoms, but it is useful to divide the
# > total electron density among them. The resulting per-atom numbers are
# > partial charges, in units of the elementary charge *e*. They are not
# > physical observables, because different partitioning schemes give different
# > values, but they are a standard and informative descriptor of where electron
# > density sits. They must sum to the total charge of the molecule.
#
# ## What distinguishes AIMNet2 from most other organic MLIPs
#
# Most machine-learned potentials for organic molecules take only the atomic
# positions and atomic numbers as input. They are therefore defined only for
# neutral, closed-shell species.
#
# AIMNet2 takes the **total charge** and the **spin multiplicity** as explicit
# inputs, and predicts the partial charge distribution as an output. This is
# what makes anions, cations and radicals accessible, and it is used directly in
# notebook 7.

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
# ## 1.1 What the model returns

# %%
ethanol = attach(build("CCO"))
from ase.data import chemical_symbols

print(f"energy        {ethanol.get_potential_energy():10.4f} eV")
print(f"largest force {np.abs(ethanol.get_forces()).max():10.4f} eV/A")
print(f"dipole moment {np.linalg.norm(ethanol.get_dipole_moment()):10.4f} e*A\n")

q = ethanol.get_charges()
print(f"{'atom':<8}{'partial charge / e':>20}")
for i, (z, qi) in enumerate(zip(ethanol.get_atomic_numbers(), q)):
    print(f"{chemical_symbols[z]}{i:<7}{qi:>20.4f}")
print(f"{'sum':<8}{q.sum():>20.4f}")

# %% [markdown]
# The oxygen carries the most negative charge and the hydroxyl hydrogen the most
# positive. This is the expected polarity of an alcohol. The model was not told
# which atom is the hydroxyl hydrogen; it follows from the geometry.
#
# ## 1.2 Element coverage
#
# The fourteen elements AIMNet2 supports include several that most organic
# neural network potentials omit. The following molecules would not be valid
# input to a model trained only on H, C, N and O.

# %%
import time

PANEL = {
    "trimethyl phosphate": "COP(=O)(OC)OC",
    "N-phenyl methanesulfonamide": "CS(=O)(=O)Nc1ccccc1",
    "trimethylsilyl methyl ether": "CO[Si](C)(C)C",
    "phenylboronic acid": "OB(O)c1ccccc1",
    "dimethyl diselenide": "C[Se][Se]C",
    "4-iodotoluene": "Ic1ccc(C)cc1",
}
t0 = time.perf_counter()
print(f"{'molecule':<30}{'atoms':>7}{'most negative atom':>22}")
print("-" * 59)
for name, smiles in PANEL.items():
    a = attach(build(smiles))
    a.get_potential_energy()
    qi = a.get_charges(); j = int(qi.argmin())
    tag = f"{chemical_symbols[a.numbers[j]]}{j}  {qi[j]:+.2f} e"
    print(f"{name:<30}{len(a):>7}{tag:>22}")
print("-" * 59)
print(f"six molecules containing P, S, Si, B, Se and I: {time.perf_counter()-t0:.1f} s")

# %% [markdown]
# ## 1.3 Total charge is an input
#
# Acetic acid and the acetate anion differ by one proton. On the model side, the
# only things that change are the removal of that proton from the atom list
# and the value of `charge`.
#
# > **Reminder — resonance.** In a carboxylate anion the negative charge is not
# > localised on one oxygen. The two C–O bonds are equivalent, and the charge is
# > shared between them. This is a consequence of the electronic structure, not
# > something imposed on the model.

# %%
from ase.optimize import LBFGS

for smiles, label in (("CC(=O)O", "acetic acid"), ("CC(=O)[O-]", "acetate")):
    a = attach(build(smiles))
    LBFGS(a, logfile=None).run(fmax=FMAX, steps=400)
    qi = a.get_charges()
    ox = [i for i, z in enumerate(a.numbers) if z == 8]
    c = [i for i, z in enumerate(a.numbers) if z == 6][1]
    print(f"{label:<13} charge {a.info['charge']:+d}   "
          f"O charges {qi[ox[0]]:+.3f} {qi[ox[1]]:+.3f} e   "
          f"C-O bonds {a.get_distance(c, ox[0]):.3f} {a.get_distance(c, ox[1]):.3f} A   "
          f"|mu| {np.linalg.norm(a.get_dipole_moment()):.2f} e*A")

# %% [markdown]
# In acetic acid the two oxygens are inequivalent: one is a carbonyl oxygen, the
# other bears the acidic proton, and the two C–O bond lengths differ by about
# 0.13 Å. In acetate both oxygens carry the same charge to within 0.005 e and
# the two C–O bond lengths are equal to within 0.001 Å. The delocalisation is
# reproduced correctly.
#
# ## 1.4 The domain of applicability
#
# A model is only meaningful within the range of chemistry it was trained on.
# For AIMNet2 the boundaries are:
#
# - the fourteen elements H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br and I,
#   and no others;
# - molecular systems, not bulk metals or surfaces;
# - **the gas phase**. No solvent is present in any model except `aimnet2-pd`.
#
# The last point has consequences that are easy to overlook. The following
# calculation makes them concrete.
#
# > **Reminder — zwitterion.** In water and in the solid state, amino acids
# > exist as zwitterions: the carboxylic acid is deprotonated and the amine is
# > protonated, giving a species with a formal positive and a formal negative
# > charge but no net charge. Solvent molecules stabilise this separation of
# > charge. In the gas phase there is nothing to stabilise it.

# %%
def proton_position(a):
    d = a.get_all_distances()
    H = [i for i, z in enumerate(a.numbers) if z == 1]
    N = [i for i, z in enumerate(a.numbers) if z == 7]
    O = [i for i, z in enumerate(a.numbers) if z == 8]
    return (min(d[i][j] for i in N for j in H),
            min(d[i][j] for i in O for j in H))

glycine = attach(build("[NH3+]CC(=O)[O-]", charge=0))
print("starting from the zwitterion:  N-H {:.3f} A   O-H {:.3f} A".format(*proton_position(glycine)))
LBFGS(glycine, logfile=None).run(fmax=FMAX, steps=500)
print("after optimisation:            N-H {:.3f} A   O-H {:.3f} A".format(*proton_position(glycine)))

# %% [markdown]
# The shortest O–H distance falls from about 1.7 Å to about 0.98 Å: the proton
# has transferred back from nitrogen to oxygen. The zwitterion is not a minimum
# on this potential energy surface.
#
# This is the correct result for an isolated molecule, and it is a correct
# statement about gas-phase glycine. It is also a reminder that the surface on
# which most chemistry of interest takes place contains solvent, and this one
# does not. The same point recurs in notebook 3.

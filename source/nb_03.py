# %% [markdown]
# # 3. Conformational analysis
#
# ## **Exercise 1 — approximately 13 minutes**
#
# Notebook 2 ended with a problem: a geometry optimisation finds only the
# minimum nearest its starting point, so a single optimised structure says
# nothing about a flexible molecule. This exercise solves that problem and uses
# the solution to answer a real chemical question.
#
# ## Concepts introduced here
#
# > **Definition — conformer.**
# > One of several structures of the *same* molecule that differ only by
# > rotation about single bonds, and which interconvert without breaking any
# > bonds. Conformers are distinct minima on the potential energy surface. They
# > are not isomers: no bonds differ, only torsion angles.
#
# > **Definition — conformational ensemble.**
# > The full set of accessible conformers together with their relative energies.
# > At room temperature a flexible molecule is not one structure but a
# > population distributed over this set.
#
# > **Definition — Boltzmann distribution.**
# > At thermal equilibrium the fraction of molecules found in a state of energy
# > *Eᵢ* is
# >
# > *pᵢ* ∝ exp(−*Eᵢ* / *kT*)
# >
# > where *kT* = 0.59 kcal/mol at 298 K. The practical consequence: a conformer
# > 1 kcal/mol above the minimum is about five times less populated; one
# > 3 kcal/mol above is about 150 times less populated and can usually be
# > ignored.
#
# ## The procedure
#
# 1. **Generate** many trial structures with different torsion angles (RDKit
#    ETKDG, a distance-geometry method with empirical torsion preferences).
# 2. **Optimise** each one with AIMNet2. Several will converge to the same
#    minimum.
# 3. **Remove duplicates**, so each distinct minimum is counted once.
# 4. **Apply the Boltzmann distribution** to obtain populations.

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
# ## Step 1 — choose your system
#
# Enter your seat number. Each system asks a specific, classical question about
# conformational preference. Read the question for yours before running
# anything.

# %%
SEAT = 0          # TODO: enter your seat number

MENU = [
 dict(id="butane", name="butane", smiles="CCCC", n=25,
      question="What is the energy difference between the anti and gauche conformers?",
      background="The textbook case. Anti places the two methyl groups as far apart as "
                 "possible; gauche brings them to within 60 degrees of each other.",
      reference="Experiment: anti lower by about 0.6 to 0.9 kcal/mol.",
      measure=("torsion", "[CH3][CH2][CH2][CH3]", (0,1,2,3), "C-C-C-C / deg")),
 dict(id="methylcyclohexane", name="methylcyclohexane", smiles="CC1CCCCC1", n=40,
      question="How large is the gap between the two chair conformers, and how "
               "does it compare with the A-value of methyl?",
      background="The single most cited conformational equilibrium in organic chemistry. "
                 "A six-membered ring adopts a chair; a substituent on it can point "
                 "roughly along the ring axis (axial) or roughly in the ring plane "
                 "(equatorial). Equatorial is the lower of the two, and the gap is\n"
                 "called the A-value. Higher conformers in your list are twist-boats,\n"
                 "which are several kcal/mol above either chair.",
      reference="Experiment: equatorial lower by 1.74 kcal/mol (the A-value of methyl).",
      measure=("none", None, None, "relative energy / kcal mol-1")),
 dict(id="ethylene_glycol", name="ethylene glycol", smiles="OCCO", n=30,
      question="Does the lowest conformer form an internal O-H...O hydrogen bond? "
               "Measure the shortest O...H distance.",
      background="Two hydroxyl groups on adjacent carbons can rotate to face each other. "
                 "If they do, one donates a hydrogen bond to the other.",
      reference="A hydrogen bond gives an H...O distance near 2.0 to 2.3 A; with no "
                "interaction the two hydroxyls point apart and the distance exceeds "
                "3.5 A.",
      measure=("hbond", None, None, "shortest H-bond contact / A")),
 dict(id="glycine", name="glycine", smiles="NCC(=O)O", n=30,
      question="How many conformers lie within 3 kcal/mol, and does the lowest one "
               "place the amine near the carbonyl?",
      background="The smallest amino acid. Even with only a few rotatable bonds it has "
                 "several accessible conformers. Two intramolecular motifs compete: a\n"
                 "bifurcated N-H...O=C contact and a shorter O-H...N contact. The\n"
                 "column below reports whichever is shortest in each conformer.",
      reference="Gas-phase glycine has several conformers within 2 kcal/mol.",
      measure=("hbond", None, None, "shortest H-bond contact / A")),
 dict(id="dichloroethane", name="1,2-dichloroethane", smiles="ClCCCl", n=25,
      question="Is anti or gauche lower, and by how much?",
      background="The same question as butane, with chlorine in place of methyl. "
                 "Compare your number directly with whoever has butane.",
      reference="Experiment: anti lower by about 1.1 kcal/mol in the gas phase.",
      measure=("torsion", "Cl[CH2][CH2]Cl", (0,1,2,3), "Cl-C-C-Cl / deg")),
 dict(id="butanediol", name="1,4-butanediol", smiles="OCCCCO", n=50,
      question="How many distinct conformers are there, and what fraction of the "
               "population does the lowest hold?",
      background="Five rotatable bonds. A useful illustration of how quickly the number "
                 "of conformers grows with flexibility.",
      reference="Expect tens of conformers and no single dominant one.",
      measure=("hbond", None, None, "shortest H-bond contact / A")),
 dict(id="ibuprofen", name="ibuprofen", smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O", n=50,
      question="How many conformers lie within 3 kcal/mol, and what fraction of the "
               "population does the single lowest one hold?",
      background="A real drug molecule with four rotatable bonds. If no single conformer "
                 "dominates, then optimising one structure was never going to describe it.",
      reference="Expect the lowest conformer to hold well under 100 percent.",
      measure=("none", None, None, "relative energy / kcal mol-1")),
 dict(id="alanine_dipeptide", name="alanine dipeptide", smiles="CC(=O)NC(C)C(=O)NC", n=50,
      question="Plot the two backbone torsions against each other. Do the populated "
               "regions match a Ramachandran plot?",
      background="The standard minimal model of a protein backbone. Its two backbone "
                 "torsion angles, phi and psi, are the axes of the Ramachandran plot that "
                 "every structural biologist knows.",
      reference="Expect the extended beta / C5 basin near phi = -150 deg to dominate in "
                "the gas phase; the alpha basin near phi = -80 deg is stabilised by water.",
      measure=("rama", "C(=O)[NX3][CX4][CX3](=O)[NX3]", None, "phi, psi / deg")),
]

entry = MENU[SEAT % len(MENU)]
print(f"System:      {entry['name']}   ({entry['smiles']})\n")
print(f"QUESTION:    {entry['question']}\n")
print(f"BACKGROUND:  {entry['background']}\n")
print(f"REFERENCE:   {entry['reference']}")

# %% [markdown]
# ## Step 2 — generate and optimise
#
# The generator is run with **three different random seeds**. Distance geometry
# is a stochastic method: a single seed can miss an entire region of
# conformational space regardless of how many structures it is asked for. Using
# several independent seeds is a more reliable search than one long run from a
# single seed.

# %%
import time
from ase.optimize import LBFGS
from rdkit.Chem import rdMolDescriptors

SEEDS = (42, 7, 1)
N_PER_SEED = entry["n"] if GPU else max(8, entry["n"] // 3)

template = Chem.AddHs(Chem.MolFromSmiles(entry["smiles"]))
numbers = [a.GetAtomicNum() for a in template.GetAtoms()]
n_rot = rdMolDescriptors.CalcNumRotatableBonds(template)

t0 = time.perf_counter()
starting = []
for seed in SEEDS:
    m = Chem.AddHs(Chem.MolFromSmiles(entry["smiles"]))
    p = AllChem.ETKDGv3(); p.randomSeed = seed; p.pruneRmsThresh = 0.4; p.numThreads = 0
    ids = AllChem.EmbedMultipleConfs(m, numConfs=N_PER_SEED, params=p)
    AllChem.MMFFOptimizeMoleculeConfs(m, numThreads=0, maxIters=300)
    starting += [m.GetConformer(c).GetPositions() for c in ids]

print(f"{n_rot} rotatable bond{'' if n_rot == 1 else 's'}; {len(starting)} "
      f"starting structures from {len(SEEDS)} seeds")

optimised, energies = [], []
for k, pos in enumerate(starting):
    a = Atoms(numbers=numbers, positions=pos)
    a.info.update(charge=0, mult=1)
    attach(a)
    if LBFGS(a, logfile=None).run(fmax=FMAX, steps=500):
        optimised.append(a); energies.append(a.get_potential_energy())
    if (k + 1) % 25 == 0:
        print(f"   {k+1}/{len(starting)} optimised   ({time.perf_counter()-t0:.0f} s)")

print(f"\n{len(optimised)} of {len(starting)} converged in {time.perf_counter()-t0:.1f} s")

# %% [markdown]
# ## Step 3 — remove duplicates
#
# Many starting structures relax into the same minimum. Two optimised structures
# are the same conformer if they have the same energy *and* the same shape.
#
# Comparing Cartesian coordinates directly does not work, because the same
# structure translated or rotated has entirely different coordinates. Instead we
# compare the sorted list of all interatomic distances, which is unchanged by
# translation, rotation and reflection.

# %%
def distance_fingerprint(atoms):
    """A rotation- and translation-invariant description of a structure."""
    d = atoms.get_all_distances()
    return np.sort(d[np.triu_indices(len(atoms), 1)])

order = np.argsort(energies)
unique, unique_e, fingerprints = [], [], []
for i in order:
    fp = distance_fingerprint(optimised[i])
    duplicate = any(abs(energies[i] - e) < 0.001 and np.abs(fp - f).max() < 0.15
                    for f, e in zip(fingerprints, unique_e))
    if not duplicate:
        unique.append(optimised[i]); unique_e.append(energies[i]); fingerprints.append(fp)

unique_e = np.array(unique_e)
rel = (unique_e - unique_e.min()) * EV2KCAL
populations = np.exp(-rel / KT_298); populations /= populations.sum()

print(f"{len(optimised)} optimised structures correspond to {len(unique)} distinct conformers\n")
print(f"{'#':>3}{'relative energy / kcal mol-1':>30}{'population':>14}")
for i in range(min(8, len(unique))):
    print(f"{i:>3}{rel[i]:>30.2f}{populations[i]:>14.1%}")
print(f"\nconformers within 3 kcal/mol: {(rel < 3).sum()}")
print(f"population of the lowest:     {populations[0]:.1%}")

# %% [markdown]
# A small number of distinct conformers is not an error. A rigid molecule has
# few minima, and how few is itself part of the answer.
#
# ## Step 4 — measure the quantity your question asks about

# %%
kind, smarts, idx, axis_label = entry["measure"]
values = None

if kind == "torsion":
    match = template.GetSubstructMatch(Chem.MolFromSmarts(smarts))
    assert match, "substructure not found"
    sel = tuple(match[k] for k in idx)
    values = [a.get_dihedral(*sel) for a in unique]
    values = [v - 360 if v > 180 else v for v in values]
    print(f"measuring the torsion through atoms {sel}")
elif kind == "hbond":
    # A hydrogen bond is X-H...Y: the hydrogen must be covalently bonded to an
    # electronegative donor X, and the acceptor Y must be far enough away THROUGH
    # THE BONDS to close a ring of at least five atoms. Both conditions matter.
    #
    # Without the first, the shortest heteroatom-to-hydrogen distance in any
    # O-CH2 group is a geminal O...H-C contact at about 2.0 A, present in every
    # conformer. Without the second, the carbonyl oxygen and the hydroxyl
    # hydrogen of a -COOH group sit 2.3 A apart across three bonds, again in
    # every conformer. Either one would give a number that looks like a hydrogen
    # bond, never changes, and therefore answers nothing.
    topological = Chem.GetDistanceMatrix(template)     # separation in bonds
    acceptors = [i for i, z in enumerate(numbers) if z in (7, 8)]
    polar_h = [i for i, z in enumerate(numbers) if z == 1
               and template.GetAtomWithIdx(i).GetNeighbors()[0].GetAtomicNum() in (7, 8)]

    def shortest(a):
        d = a.get_all_distances()
        contacts = [d[y][h] for y in acceptors for h in polar_h
                    if topological[h][y] >= 4]
        return min(contacts) if contacts else float("nan")

    values = [shortest(a) for a in unique]
elif kind == "rama":
    match = template.GetSubstructMatch(Chem.MolFromSmarts(smarts))
    assert match, "backbone substructure not found"
    c_prev, _, n, ca, c, _, n_next = match
    phi = [a.get_dihedral(c_prev, n, ca, c) for a in unique]
    psi = [a.get_dihedral(n, ca, c, n_next) for a in unique]
    phi = [p - 360 if p > 180 else p for p in phi]
    psi = [p - 360 if p > 180 else p for p in psi]
    values = list(zip(phi, psi))
else:
    values = list(rel)

print(f"\n{'#':>3}{'rel E':>9}{'population':>13}{axis_label:>30}")
for i in range(min(10, len(unique))):
    v = values[i]
    s = f"{v[0]:.0f}, {v[1]:.0f}" if isinstance(v, tuple) else f"{v:.2f}"
    print(f"{i:>3}{rel[i]:>9.2f}{populations[i]:>13.1%}{s:>30}")

# %% [markdown]
# ## Step 5 — look at it

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 4.2))
if kind == "rama":
    sc = ax.scatter([v[0] for v in values], [v[1] for v in values],
                    c=rel, cmap="viridis_r", s=90, ec="black", lw=0.4)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xticks(range(-180, 181, 90)); ax.set_yticks(range(-180, 181, 90))
    ax.set_xlabel("phi / degrees"); ax.set_ylabel("psi / degrees")
    fig.colorbar(sc, ax=ax, label="relative energy / kcal mol$^{-1}$")
elif kind == "none":
    ax.bar(range(len(rel)), rel, color="#C41230")
    ax.set_xlabel("conformer index"); ax.set_ylabel("relative energy / kcal mol$^{-1}$")
else:
    ax.scatter(values, rel, c=populations, cmap="viridis_r", s=100, ec="black", lw=0.4)
    ax.set_xlabel(axis_label); ax.set_ylabel("relative energy / kcal mol$^{-1}$")
    if kind == "torsion":
        ax.set_xlim(-180, 180); ax.set_xticks(range(-180, 181, 60))
ax.set_title(entry["name"])
fig.tight_layout(); plt.show()

# %% [markdown]
# ## Step 6 — your answer
#
# Write one or two sentences answering the question posed at the top, quoting
# the numbers you obtained, and stating whether they agree with the reference.

# %%
ANSWER = ""      # TODO: your answer, in complete sentences

assert len(ANSWER) > 40, "Please write a complete answer."
print(f"seat {SEAT}\t{entry['id']}\t{len(unique)} conformers\t"
      f"spread {rel.max():.2f} kcal/mol\tlowest {populations[0]:.1%}\n{ANSWER}")

# %% [markdown]
# ## A note on what limits this result
#
# Two caveats apply to every number produced above.
#
# **Sampling.** The conformers found are those the generator happened to
# produce. If a low-lying conformer requires an unusual combination of torsions,
# a short search can miss it entirely, and the calculated populations will be
# wrong without any warning. Using several independent random seeds, as done
# here, reduces but does not eliminate this risk.
#
# **The gas phase.** These are isolated molecules. Conformers stabilised by an
# internal hydrogen bond are favoured more strongly here than they would be in
# water, where solvent molecules compete for the same hydrogen bond. Systems in
# this exercise that form internal hydrogen bonds will therefore show a stronger
# preference than is observed in solution.

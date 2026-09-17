# %% [markdown]
# # 7. Chemical reactivity
#
# ## **Exercise 3 — approximately 13 minutes**
#
# Every previous notebook asked about structure and energy. This one asks a
# different kind of question: given a molecule, **where will it react, and how
# readily?** The answer is obtained without computing a single transition state.
#
# This exercise depends on a capability that most machine-learned potentials do
# not have. It requires the energy and the charge distribution of a molecule
# with one electron added and with one electron removed. AIMNet2 takes the total
# charge and the spin multiplicity as inputs, so those states are directly
# accessible.
#
# ## Concepts introduced here
#
# > **Definition — ionisation potential (*I*).**
# > The energy required to remove one electron: *I* = *E*(*N*−1) − *E*(*N*),
# > where *N* is the number of electrons. Computed here at the geometry of the
# > neutral molecule, which is called the *vertical* ionisation potential.
#
# > **Definition — electron affinity (*A*).**
# > The energy released on adding one electron: *A* = *E*(*N*) − *E*(*N*+1).
# > A negative value means the anion is higher in energy than the neutral
# > molecule, i.e. the isolated molecule does not bind an extra electron. This is
# > the normal situation in the gas phase for most closed-shell organics.
#
# > **Definition — electronic chemical potential (*μ*) and chemical hardness (*η*).**
# > *μ* = −(*I* + *A*)/2 measures the escaping tendency of the electrons: a more
# > negative *μ* means the molecule holds its electrons more tightly.
# > *η* = *I* − *A* measures resistance to change in the electron number. A
# > large *η* is a "hard" molecule, a small *η* a "soft" one.
#
# > **Definition — global electrophilicity index (*ω*).**
# > *ω* = *μ*² / (2*η*), introduced by Parr. It quantifies how much the energy of
# > a molecule is lowered when it accepts electron density from a generic donor.
# > A larger *ω* means a stronger electrophile. It is a *single number for the
# > whole molecule*, so it ranks reactivity but says nothing about position.
#
# > **Definition — Fukui function.**
# > The position-resolved counterpart. Condensed onto atom *k*:
# >
# > *f*⁺(*k*) = *q*(*k*, *N*) − *q*(*k*, *N*+1), i.e. where added electron density goes.
# > Large *f*⁺ marks the site most susceptible to **nucleophilic attack**.
# >
# > *f*⁻(*k*) = *q*(*k*, *N*−1) − *q*(*k*, *N*), i.e. where electron density is most
# > readily removed. Large *f*⁻ marks the site most susceptible to
# > **electrophilic attack**.
# >
# > Here *q* is the partial charge on atom *k*. These are the frontier-orbital
# > ideas of Fukui expressed in terms of quantities the model outputs directly.
#
# ## What you will do
#
# 1. Optimise the neutral molecule.
# 2. Evaluate it at *N*, *N*+1 and *N*−1 electrons, at that same geometry.
# 3. From the three energies obtain *I*, *A*, *μ*, *η* and *ω*.
# 4. From the three charge distributions obtain *f*⁺ and *f*⁻ on every atom.
# 5. Identify the reactive site and check it against known chemistry.

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
    import aimnet, rdkit, ase, py3Dmol  # noqa: F401
except ImportError:
    _pip("aimnet[ase]", "rdkit", "py3Dmol", "warp-lang<1.18")
import numpy as np
import torch
from aimnet.calculators import AIMNet2Calculator, AIMNet2ASE

GPU = torch.cuda.is_available()
print(f"Python {sys.version.split()[0]}   PyTorch {torch.__version__}   GPU available: {GPU}")

_ = AIMNet2Calculator("aimnet2-nse")          # downloads parameters on first use
print("Model loaded.")
# %% [markdown]
# ### Three helper functions
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

import py3Dmol
from ase.data import chemical_symbols

def show(structures, labels=None, animate=False, loop="backAndForth",
         width=380, height=300):
    """
    Interactive three-dimensional view. Drag to rotate, scroll to zoom.

    `structures` is one Atoms object or a list of them. A list is drawn side
    by side or, with `animate=True`, played as a film. `labels` is a list of
    per-atom strings, one list per structure, printed on the atoms.

    The viewer is 3Dmol.js, fetched from the web when the cell runs, so it
    needs a network connection. Nothing in this tutorial depends on it: every
    result is also printed as text.
    """
    if isinstance(structures, Atoms):
        structures = [structures]
        labels = None if labels is None else [labels]

    def xyz(a):
        return f"{len(a)}\n\n" + "".join(
            f"{chemical_symbols[z]} {x:.5f} {y:.5f} {zz:.5f}\n"
            for z, (x, y, zz) in zip(a.numbers, a.positions))

    style = {"stick": {"radius": 0.14}, "sphere": {"scale": 0.24}}
    if animate:
        v = py3Dmol.view(width=width, height=height)
        v.addModelsAsFrames("".join(xyz(a) for a in structures), "xyz")
        v.setStyle(style)
        v.animate({"loop": loop, "interval": 80})
    else:
        n = len(structures)
        v = py3Dmol.view(width=width * n, height=height, viewergrid=(1, n))
        for k, a in enumerate(structures):
            v.addModel(xyz(a), "xyz", viewer=(0, k))
            v.setStyle(style, viewer=(0, k))
            for text, (x, y, z) in zip(labels[k] if labels else [], a.positions):
                if str(text):
                    v.addLabel(str(text), {"position": {"x": x, "y": y, "z": z},
                                           "fontSize": 11, "backgroundOpacity": 0.55,
                                           "inFront": True}, viewer=(0, k))
    v.zoomTo()
    v.show()

EV2KCAL = 23.060548     # kcal per mol, per eV
KT_298 = 0.5924         # kT at 298.15 K, in kcal/mol
FMAX = 0.02             # force convergence threshold, eV per Angstrom

# %% [markdown]
# ## The reactivity calculation
#
# Note the multiplicities. Removing an electron from a closed-shell molecule, or
# adding one to it, leaves an unpaired electron, so the ion is a **doublet**,
# `mult = 2`.
# Requesting `mult = 1` for these species would be physically wrong. The model
# `aimnet2-nse` is the family member trained for open-shell systems.

# %%
from ase.data import chemical_symbols

MODEL = "aimnet2-nse"

def reactivity(atoms):
    """Global descriptors and condensed Fukui functions, at fixed geometry."""
    def evaluate(charge, mult):
        a = atoms.copy()
        a.info.update(charge=charge, mult=mult)
        attach(a, MODEL)
        return a.get_potential_energy(), np.asarray(a.get_charges())

    E_neutral, q_neutral = evaluate(0, 1)      # N electrons,   closed shell
    E_anion,   q_anion   = evaluate(-1, 2)     # N+1 electrons, doublet
    E_cation,  q_cation  = evaluate(+1, 2)     # N-1 electrons, doublet

    I = E_cation - E_neutral                   # ionisation potential, eV
    A = E_neutral - E_anion                    # electron affinity, eV
    mu = -(I + A) / 2                          # chemical potential
    eta = I - A                                # chemical hardness
    omega = mu ** 2 / (2 * eta)                # electrophilicity index

    f_plus = q_neutral - q_anion               # nucleophilic attack site
    f_minus = q_cation - q_neutral             # electrophilic attack site
    return dict(I=I, A=A, mu=mu, eta=eta, omega=omega,
                q=q_neutral, f_plus=f_plus, f_minus=f_minus)

print("Reactivity analysis defined.")

# %% [markdown]
# ## Step 1 — choose your molecule

# %%
SEAT = 0          # TODO: enter your seat number

MENU = [
 dict(id="mvk", name="methyl vinyl ketone", smiles="CC(=O)C=C", kind="carbonyl",
      question="Which carbon is the most electrophilic, and is it the one you would "
               "predict from the resonance structures?",
      background="An alpha,beta-unsaturated ketone, i.e. a Michael acceptor. Conjugation "
                 "places positive character on the carbon beta to the carbonyl, which is "
                 "where a nucleophile adds in a conjugate (1,4) addition.",
      reference="Expect the largest f+ on the terminal CH2 carbon, not on the carbonyl carbon."),
 dict(id="acrolein", name="acrolein", smiles="C=CC=O", kind="carbonyl",
      question="Which carbon carries the largest f+? Compare your electrophilicity index "
               "with whoever has methyl vinyl ketone.",
      background="The simplest alpha,beta-unsaturated aldehyde. Methyl vinyl ketone is "
                 "the same system with an electron-donating methyl group attached.",
      reference="Expect the terminal CH2. Expect acrolein to be the stronger electrophile "
                "of the two, because it lacks the donating methyl."),
 dict(id="acrylonitrile", name="acrylonitrile", smiles="C=CC#N", kind="carbonyl",
      question="Which carbon carries the largest f+, and how does omega compare with the "
               "other Michael acceptors in the room?",
      background="A nitrile in place of a carbonyl. The C#N group withdraws electron "
                 "density strongly.",
      reference="Expect the terminal CH2, and a large omega."),
 dict(id="acetone", name="acetone", smiles="CC(C)=O", kind="carbonyl",
      question="Which atom is most electrophilic here, and why is the answer different "
               "from the conjugated systems?",
      background="A simple ketone with no conjugated double bond. There is no beta-carbon "
                 "for charge to delocalise onto.",
      reference="Expect the carbonyl carbon itself, and a notably smaller omega than any "
                "of the conjugated acceptors."),
 dict(id="phenol", name="phenol", smiles="Oc1ccccc1", kind="arene", anchor="[OX2H]",
      question="At which ring positions is f- largest: ortho, meta or para?",
      background="Electrophilic aromatic substitution. An -OH group donates electron "
                 "density into the ring through resonance, and the classical result is "
                 "that it directs incoming electrophiles to the ortho and para positions.",
      reference="Expect ortho and para to exceed meta clearly."),
 dict(id="aniline", name="aniline", smiles="Nc1ccccc1", kind="arene", anchor="[NX3H2]",
      question="At which ring positions is f- largest, and how does the ionisation "
               "potential compare with benzene at 9.4 eV?",
      background="An -NH2 group is an even stronger pi-donor than -OH.",
      reference="Expect ortho and para, and an ionisation potential well below benzene's."),
 dict(id="anisole", name="anisole", smiles="COc1ccccc1", kind="arene", anchor="[OX2]([CH3])",
      question="At which ring positions is f- largest?",
      background="The methyl ether of phenol. The oxygen still donates into the ring.",
      reference="Expect ortho and para."),
 dict(id="nitrobenzene", name="nitrobenzene", smiles="O=[N+]([O-])c1ccccc1", kind="arene",
      anchor="[N+](=O)[O-]",
      question="At which ring positions is f- largest? Compare your answer carefully with "
               "the classical result. This one is the hard case.",
      background="A nitro group withdraws electron density strongly and deactivates the "
                 "ring. The classical result is meta direction.",
      reference="Experiment: about 93 percent meta substitution. Read the closing section "
                "of this notebook before drawing a conclusion."),
]

entry = MENU[SEAT % len(MENU)]
print(f"Molecule:    {entry['name']}   ({entry['smiles']})\n")
print(f"QUESTION:    {entry['question']}\n")
print(f"BACKGROUND:  {entry['background']}\n")
print(f"REFERENCE:   {entry['reference']}")

# %% [markdown]
# ## Step 2 — optimise, then evaluate the three charge states

# %%
import time
from ase.optimize import LBFGS

molecule = attach(build(entry["smiles"]), MODEL)
LBFGS(molecule, logfile=None).run(fmax=FMAX, steps=600)

t0 = time.perf_counter()
r = reactivity(molecule)
print(f"three single-point evaluations in {time.perf_counter()-t0:.1f} s\n")
print(f"  ionisation potential   I     = {r['I']:7.2f} eV")
print(f"  electron affinity      A     = {r['A']:7.2f} eV")
print(f"  chemical potential     mu    = {r['mu']:7.2f} eV")
print(f"  chemical hardness      eta   = {r['eta']:7.2f} eV")
print(f"  electrophilicity index omega = {r['omega']:7.2f} eV")

if r["A"] < 0:
    print("\nThe electron affinity is negative: this isolated molecule does not bind an")
    print("extra electron. That is normal in the gas phase, and it is why eta is large.")

# %% [markdown]
# ## Step 3 — where the molecule reacts
#
# The Fukui values are printed for every heavy atom. Hydrogens are omitted:
# they are rarely the site of attack and they clutter the table.

# %%
numbers = molecule.get_atomic_numbers()
heavy = [i for i, z in enumerate(numbers) if z > 1]

print(f"{'atom':<8}{'q / e':>10}{'f+':>10}{'f-':>10}")
print("-" * 38)
for i in heavy:
    print(f"{chemical_symbols[numbers[i]]}{i:<7}{r['q'][i]:>10.3f}"
          f"{r['f_plus'][i]:>10.3f}{r['f_minus'][i]:>10.3f}")

site_plus = max(heavy, key=lambda i: r["f_plus"][i])
site_minus = max(heavy, key=lambda i: r["f_minus"][i])
print(f"\nlargest f+ : {chemical_symbols[numbers[site_plus]]}{site_plus}"
      f"  ({r['f_plus'][site_plus]:.3f})  -> most susceptible to nucleophilic attack")
print(f"largest f- : {chemical_symbols[numbers[site_minus]]}{site_minus}"
      f"  ({r['f_minus'][site_minus]:.3f})  -> most susceptible to electrophilic attack")

# %% [markdown]
# *f*⁺ printed on each heavy atom.

# %%
show(molecule, labels=[f"{r['f_plus'][i]:.2f}" if numbers[i] > 1 else ""
                       for i in range(len(numbers))])

# %% [markdown]
# ## Step 4 — resolve the answer by position
#
# For a carbonyl system, compare the carbons. For an aromatic ring, group the
# ring carbons into ortho, meta and para relative to the substituent.

# %%
if entry["kind"] == "carbonyl":
    carbons = [i for i in heavy if numbers[i] == 6]
    print("carbon atoms only, ranked by f+ (susceptibility to nucleophilic attack):\n")
    for i in sorted(carbons, key=lambda i: -r["f_plus"][i]):
        print(f"  C{i:<4}{r['f_plus'][i]:>8.3f}  " + "#" * int(round(r['f_plus'][i] * 90)))
else:
    template = Chem.AddHs(Chem.MolFromSmiles(entry["smiles"]))
    hit = template.GetSubstructMatch(Chem.MolFromSmarts(entry["anchor"]))
    ipso = [n.GetIdx() for n in template.GetAtomWithIdx(hit[0]).GetNeighbors()
            if n.GetIsAromatic()][0]
    dmat = Chem.GetDistanceMatrix(template)
    labels = {}
    for a in template.GetAtoms():
        if a.GetIsAromatic() and a.GetAtomicNum() == 6:
            labels[a.GetIdx()] = {0: "ipso", 1: "ortho", 2: "meta", 3: "para"}[
                int(dmat[ipso][a.GetIdx()])]
    grouped = {}
    for i, lab in labels.items():
        grouped.setdefault(lab, []).append(r["f_minus"][i])
    print("ring positions, mean f- (susceptibility to electrophilic attack):\n")
    for lab in ("ipso", "ortho", "meta", "para"):
        if lab in grouped:
            m = float(np.mean(grouped[lab]))
            print(f"  {lab:<7}{m:>8.3f}  " + "#" * int(round(m * 110)))
    ranked = {k: float(np.mean(v)) for k, v in grouped.items() if k != "ipso"}
    print(f"\npredicted site of electrophilic attack: "
          f"{max(ranked, key=ranked.get)}")

# %% [markdown]
# ## Step 5 — your answer

# %%
ANSWER = ""     # TODO: Which site does the calculation identify, does it agree with
                #       the classical prediction, and what is your omega? Two or three sentences.

assert len(ANSWER) > 40, "Please write a complete answer."
print(f"seat {SEAT}\t{entry['id']}\tomega {r['omega']:.2f} eV\t"
      f"I {r['I']:.2f}\tA {r['A']:.2f}\n{ANSWER}")

# %% [markdown]
# ## When the room reports back
#
# Two patterns should be visible across the whole set of results.
#
# **The electrophilicity index ranks the Michael acceptors correctly.** Acetone,
# which has no conjugated double bond, is the weakest of the four carbonyl
# systems. The conjugated acceptors are all stronger, and among them the
# ordering follows the strength of the electron-withdrawing group. (The
# activated arenes on seats 4 to 6 have lower *ω* still, which is the point:
# they are nucleophiles, not electrophiles.)
#
# **The Fukui functions locate the site, which *ω* cannot.** For every
# conjugated acceptor the largest *f*⁺ falls on the carbon *β* to the carbonyl,
# not on the carbonyl carbon itself. That is the site of conjugate addition, and
# it is recovered here from partial charges alone, with no orbital analysis and
# no transition state.
#
# ## The limitation, stated plainly
#
# For the **activated** aromatic rings (phenol, aniline, anisole) the
# condensed *f*⁻ correctly places ortho and para above meta.
#
# For **deactivated** rings such as nitrobenzene, the same descriptor does not
# reproduce the classical meta preference reliably. It tends to favour the para
# position instead.
#
# This is a known limitation, and the reason is worth understanding. Condensed
# Fukui functions measure how the *ground-state* electron density redistributes
# when an electron is added or removed. Regioselectivity in electrophilic
# aromatic substitution is determined by the relative stabilities of the
# *σ*-complex intermediates, which is a different quantity. For strongly
# activated rings the two correlate well; for deactivated rings they do not.
#
# The useful conclusion is not that the method is unreliable, but that it is a
# **screening** tool: fast enough to rank hundreds of molecules, and correct
# often enough to be worth doing, provided its failure mode is known. Where the
# answer matters, compute the *σ*-complex energies explicitly, using the
# transition-state machinery of notebook 5.

# %% [markdown]
# # 5. Reactions and transition states
#
# *Demonstration notebook.*
#
# So far every structure examined has been a minimum. Chemistry is about the
# paths between minima.
#
# ## Concepts introduced here
#
# > **Definition — transition state.**
# > The highest-energy point along the lowest-energy path connecting reactants to
# > products. Mathematically it is a **first-order saddle point** on the
# > potential energy surface: a maximum along exactly one direction, and a
# > minimum along all the others.
#
# > **Definition — activation barrier.**
# > The energy of the transition state relative to the reactants. It controls
# > the rate: at room temperature, each additional 1.4 kcal/mol of barrier slows
# > a reaction roughly tenfold.
#
# > **Definition — Hessian.**
# > The matrix of second derivatives of the energy with respect to the atomic
# > coordinates, ∂²*E*/∂*Rᵢ*∂*Rⱼ*. It describes the curvature of the surface.
# > Its eigenvalues, after mass-weighting, give the vibrational frequencies.
#
# > **Definition — imaginary frequency.**
# > A frequency obtained from a *negative* Hessian eigenvalue, i.e. a direction
# > of negative curvature. It is reported as a negative number by convention. A
# > minimum has none. A transition state has **exactly one**, and the
# > corresponding motion is the reaction itself.
#
# ## The example
#
# Chloride attacking chloromethane: the S<sub>N</sub>2 reaction, and the first mechanism
# most chemists learn to draw.
#
# > **Reminder — S<sub>N</sub>2.** A nucleophile attacks a carbon from the side opposite
# > the leaving group. The carbon passes through a planar arrangement of its
# > three remaining substituents and emerges with its configuration inverted.

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

try:
    from sella import Sella
except ImportError:
    _pip("sella")
    from sella import Sella
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
# ## 5.1 Choosing the model
#
# AIMNet2 is a family of models, not one model. Two are relevant here.
#
# | Model | Elements | Charge | Intended for |
# |---|---|---|---|
# | `aimnet2-rxn` | H, C, N, O only | neutral only | reaction paths at scale |
# | `aimnet2-nse` | all fourteen | any charge, any multiplicity | open-shell and charged systems |
#
# The system here contains chlorine and carries a charge of −1, so
# `aimnet2-rxn` is excluded on both counts. It does not fail silently.

# %%

chloride_complex = Atoms("CClClHHH", positions=[
    [0, 0, 0], [1.80, 0, 0], [-3.40, 0, 0],
    [-0.36, 1.03, 0], [-0.36, -0.515, 0.892], [-0.36, -0.515, -0.892]])
chloride_complex.info.update(charge=-1, mult=1)

try:
    test = chloride_complex.copy(); test.info.update(charge=-1, mult=1)
    test.calc = AIMNet2ASE(AIMNet2Calculator("aimnet2-rxn"), charge=-1, mult=1)
    test.get_potential_energy()
    print("No error raised. Check the installed version.")
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    print("\nThe model refused the input rather than returning a plausible but wrong number.")

# %% [markdown]
# ## 5.2 Locating the transition state
#
# An ordinary optimiser moves downhill and finds minima. A transition state is a
# maximum in one direction, so a different algorithm is required: one that
# identifies the direction of negative curvature from the Hessian and climbs
# along it while minimising along all others.
#
# This is where AIMNet2 differs usefully from most machine-learned potentials.
# It provides **analytic second derivatives**, obtained by differentiating the
# network twice by automatic differentiation. The alternative, finite differences, requires
# 6*N* additional gradient evaluations and a step size that must be chosen.

# %%
import time

# A reasonable guess: the two C-Cl distances equal, the three hydrogens planar.
ts = Atoms("CClClHHH", positions=[
    [0, 0, 0], [2.32, 0, 0], [-2.32, 0, 0],
    [0, 1.07, 0], [0, -0.535, 0.927], [0, -0.535, -0.927]])
ts.info.update(charge=-1, mult=1)
ts.calc = AIMNet2ASE(AIMNet2Calculator("aimnet2-nse"), charge=-1, mult=1)

t0 = time.perf_counter()
search = Sella(ts, order=1, internal=True,
               hessian_function=ts.calc.get_hessian, logfile=None)
converged = search.run(fmax=0.03, steps=120)

print(f"converged: {converged} in {search.get_number_of_steps()} steps, "
      f"{time.perf_counter()-t0:.1f} s")
print(f"largest remaining force: {np.abs(ts.get_forces()).max():.4f} eV/A")
print(f"C-Cl distances: {ts.get_distance(0,1):.3f} and {ts.get_distance(0,2):.3f} A")

# %%
show(ts)          # the three hydrogens in a plane, a chlorine on either side

# %% [markdown]
# `order=1` instructs the algorithm to seek a first-order saddle point. The two
# C–Cl distances converge to the same value, as symmetry requires for this
# reaction.
#
# ## 5.3 Confirming that it is a transition state
#
# A structure with zero forces could be a minimum, a transition state, or a
# higher-order saddle point. Only the Hessian distinguishes them.
#
# The procedure has three steps:
#
# 1. **Mass-weight** the Hessian. A heavy atom moves less than a light one for
#    the same force, so each element is divided by √(*mᵢmⱼ*).
# 2. **Project out** the six motions that do not change the internal geometry:
#    three translations of the whole molecule and three rotations. These
#    correspond to zero curvature and would otherwise appear as six spurious
#    near-zero frequencies. A molecule of *N* atoms therefore has 3*N* − 6
#    genuine vibrations (3*N* − 5 if linear).
# 3. **Diagonalise**, and convert eigenvalues to frequencies.

# %%
from ase.data import atomic_masses

def normal_modes(hessian, numbers, positions):
    """Vibrational frequencies in cm-1, with translations and rotations removed.

    Returns (frequencies, cartesian displacements, number of rigid-body modes).
    Imaginary frequencies are returned as negative numbers, following the
    convention used by every quantum chemistry program.
    """
    numbers = np.asarray(numbers)
    masses = np.array([atomic_masses[int(z)] for z in numbers])
    n = len(numbers)
    H = np.asarray(hessian, float).reshape(3 * n, 3 * n)
    H = 0.5 * (H + H.T)                                  # enforce exact symmetry
    w = np.repeat(1.0 / np.sqrt(masses), 3)
    H_mw = H * np.outer(w, w)                            # step 1: mass-weight

    # step 2: build and remove the rigid-body subspace
    sm = np.sqrt(masses)
    centre = (positions * masses[:, None]).sum(0) / masses.sum()
    r = positions - centre
    D = np.zeros((6, 3 * n))
    for i in range(n):
        for c in range(3):
            D[c, 3 * i + c] = sm[i]                      # translations
        x, y, z = r[i]
        D[3, 3*i:3*i+3] = sm[i] * np.array([0.0, -z, y])  # rotations
        D[4, 3*i:3*i+3] = sm[i] * np.array([z, 0.0, -x])
        D[5, 3*i:3*i+3] = sm[i] * np.array([-y, x, 0.0])
    _, s, Vt = np.linalg.svd(D, full_matrices=False)
    basis = Vt[s > 1e-8]
    n_rigid = basis.shape[0]
    P = np.eye(3 * n) - basis.T @ basis

    values, vectors = np.linalg.eigh(P @ H_mw @ P)       # step 3: diagonalise
    keep = np.argsort(np.abs(values))[n_rigid:]
    keep = keep[np.argsort(values[keep])]
    # 0.0646594 eV per sqrt(eV / A^2 / amu); 8065.54 cm-1 per eV
    freqs = 0.0646594 * np.sqrt(np.abs(values[keep])) * np.sign(values[keep]) * 8065.543937
    disp = (vectors[:, keep] * w[:, None]).T.reshape(-1, n, 3)
    return freqs, disp, n_rigid

def along_mode(atoms, d, amplitude=0.5, n=12):
    """Frames of the structure displaced along one normal mode, for animation."""
    d = np.asarray(d) / np.abs(d).max()
    frames = []
    for s in np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        f = atoms.copy()
        f.set_positions(atoms.get_positions() + amplitude * s * d)
        frames.append(f)
    return frames

H = ts.calc.get_hessian(ts)
numbers = ts.get_atomic_numbers()
freqs, disp, n_rigid = normal_modes(H, numbers, ts.get_positions())

print(f"Hessian shape {np.asarray(H).shape}, units eV / Angstrom^2")
print(f"{len(numbers)} atoms -> 3N - {n_rigid} = {len(freqs)} vibrations\n")
for k, f in enumerate(freqs):
    marker = "   <-- imaginary" if f < -50 else ""
    print(f"  mode {k+1:>2}  {f:>9.0f} cm-1{marker}")

n_imaginary = int((freqs < -50).sum())
print(f"\n{n_imaginary} imaginary mode(s): "
      + {0: "a minimum", 1: "a first-order saddle point, i.e. a transition state"}
        .get(n_imaginary, "a higher-order saddle point, not a transition state"))

# %% [markdown]
# ## 5.4 What the imaginary mode is
#
# The displacement pattern of the imaginary mode is the reaction coordinate: the
# motion that carries reactants into products.

# %%
from ase.data import chemical_symbols

magnitudes = np.linalg.norm(disp[0] / np.abs(disp[0]).max(), axis=1)
print("displacement of each atom in the imaginary mode, normalised:\n")
for i in np.argsort(-magnitudes):
    print(f"  {chemical_symbols[int(numbers[i])]}{i:<3}{magnitudes[i]:>7.2f}  "
          + "#" * int(round(magnitudes[i] * 30)))

# %% [markdown]
# The same mode as a film: the structure displaced back and forth along the
# imaginary mode. Forward is the reaction, backward is the reverse reaction.

# %%
show(along_mode(ts, disp[0], amplitude=0.7), animate=True, loop="forward")

# %% [markdown]
# The carbon carries almost all of the motion; the two chlorines move little.
# What the mode describes is the carbon passing through the plane of its three
# hydrogens while one C–Cl bond lengthens and the other shortens. That is
# precisely the inversion of configuration that defines an S<sub>N</sub>2 reaction.
#
# ## 5.5 Beyond a single saddle point
#
# Locating a transition state from a guessed structure works when the guess is
# good. For an unfamiliar reaction it usually is not, and a chain-of-states
# method is used instead: a series of structures is placed between reactants and
# products and relaxed together.
#
# AIMNet2 registers itself with **pysisyphus**, which implements these methods:
#
# ```yaml
# geom: {type: cart, fn: [reactant.xyz, product.xyz]}
# calc: {type: aimnet, model: aimnet2-nse, charge: -1, mult: 1}
# cos:  {type: neb, nimages: 11, climb: True}
# opt:  {type: lbfgs, max_cycles: 200}
# ```
#
# ```bash
# pip install "aimnet[pysis]"
# aimnet2pysis neb.yaml
# ```
#
# ## Practical notes
#
# Three restrictions apply to Hessians, and all three raise an exception rather
# than returning a wrong answer:
#
# - `compile_model=True` cannot be combined with a Hessian request;
# - Hessians are available for a single molecule at a time, and memory grows as
#   *N*²;
# - `aimnet2-rxn` rejects any charged input.
#
# ---
#
# **The next notebook applies exactly the same machinery to a minimum instead of
# a saddle point.** At a minimum every frequency is real, and the set of them,
# weighted by how strongly each changes the molecular dipole, is an infrared
# spectrum.

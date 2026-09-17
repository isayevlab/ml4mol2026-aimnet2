# %% [markdown]
# # 6. Infrared spectra
#
# ## **Exercise 2 — approximately 13 minutes**
#
# Notebook 5 computed a Hessian at a saddle point and found one imaginary
# frequency. This exercise computes a Hessian at a **minimum**, where every
# frequency is real, and turns the result into a spectrum that can be compared
# with a measurement.
#
# ## Concepts introduced here
#
# > **Definition — normal mode.**
# > A collective vibration in which every atom oscillates at the same frequency
# > and passes through its equilibrium position at the same instant. Any
# > vibrational motion of the molecule can be written as a sum of normal modes.
# > A non-linear molecule of *N* atoms has 3*N* − 6 of them.
#
# > **Definition — harmonic approximation.**
# > The assumption that the potential energy near a minimum is quadratic in the
# > displacements. Under this assumption the normal-mode frequencies follow
# > directly from the Hessian. The approximation is good for small displacements
# > and systematically overestimates the frequency of X–H stretches, which are
# > strongly anharmonic, by roughly 3 to 5 percent.
#
# > **Definition — infrared selection rule.**
# > A vibration absorbs infrared radiation only if it changes the molecular
# > dipole moment. The intensity is proportional to |∂**μ**/∂*Q*|², where *Q* is
# > the normal coordinate. A mode that moves atoms without changing the dipole is
# > **infrared inactive** and does not appear in the spectrum, no matter how
# > large its amplitude.
#
# ## What you will do
#
# 1. Optimise your molecule.
# 2. Compute the Hessian and extract the normal modes.
# 3. Compute the dipole derivative along each mode to obtain intensities.
# 4. Broaden the result into a spectrum and assign the bands.
# 5. Compare band positions with experimental values supplied in the notebook.

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

_ = AIMNet2Calculator("aimnet2")          # downloads parameters on first use
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
# ## The vibrational analysis
#
# Identical to notebook 5, with two additions: intensities, and frames for
# watching a mode.

# %%
from ase.data import atomic_masses, chemical_symbols

def normal_modes(hessian, numbers, positions):
    """Frequencies in cm-1 with translations and rotations projected out."""
    numbers = np.asarray(numbers)
    masses = np.array([atomic_masses[int(z)] for z in numbers])
    n = len(numbers)
    H = np.asarray(hessian, float).reshape(3 * n, 3 * n)
    H = 0.5 * (H + H.T)
    w = np.repeat(1.0 / np.sqrt(masses), 3)
    H_mw = H * np.outer(w, w)

    sm = np.sqrt(masses)
    centre = (positions * masses[:, None]).sum(0) / masses.sum()
    r = positions - centre
    D = np.zeros((6, 3 * n))
    for i in range(n):
        for c in range(3):
            D[c, 3 * i + c] = sm[i]
        x, y, z = r[i]
        D[3, 3*i:3*i+3] = sm[i] * np.array([0.0, -z, y])
        D[4, 3*i:3*i+3] = sm[i] * np.array([z, 0.0, -x])
        D[5, 3*i:3*i+3] = sm[i] * np.array([-y, x, 0.0])
    _, s, Vt = np.linalg.svd(D, full_matrices=False)
    basis = Vt[s > 1e-8]; n_rigid = basis.shape[0]
    P = np.eye(3 * n) - basis.T @ basis

    values, vectors = np.linalg.eigh(P @ H_mw @ P)
    keep = np.argsort(np.abs(values))[n_rigid:]
    keep = keep[np.argsort(values[keep])]
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

def ir_intensities(atoms, disp, step=0.01):
    """
    Relative infrared intensities.

    The dipole moment is computed at displaced geometries to obtain the
    derivative of the dipole with respect to each Cartesian coordinate, then
    projected onto each normal mode. This requires 6N dipole evaluations, which
    is affordable because each one is a single AIMNet2 call.

    The partial charges are recomputed at every displaced geometry. Reusing the
    equilibrium charges would drop the charge-flux term, which dominates the
    intensity of polar bonds such as C=O.
    """
    n = len(atoms)
    reference = atoms.get_positions().copy()
    dmu = np.zeros((n, 3, 3))
    for i in range(n):
        for c in range(3):
            accumulated = np.zeros(3)
            for sign in (+1, -1):
                p = reference.copy(); p[i, c] += sign * step
                atoms.set_positions(p)
                atoms.get_potential_energy()            # forces a fresh evaluation
                q = np.asarray(atoms.get_charges())
                accumulated += sign * (q[:, None] * p).sum(axis=0)
            dmu[i, c] = accumulated / (2 * step)
    atoms.set_positions(reference)
    values = np.array([float(np.einsum("ic,icx->x", d, dmu) @ np.einsum("ic,icx->x", d, dmu))
                       for d in disp])
    return values / (values.max() if values.max() > 0 else 1.0)

def spectrum(freqs, intensities, lo=400, hi=4000, width=18, npts=1600):
    """Broaden the discrete lines into a curve, as instrument resolution does."""
    x = np.linspace(lo, hi, npts)
    y = np.zeros_like(x)
    for f, a in zip(freqs, intensities):
        if f > 0:
            y += a * width**2 / ((x - f)**2 + width**2)
    return x, y / (y.max() if y.max() > 0 else 1.0)

print("Vibrational analysis defined.")

# %% [markdown]
# ## Step 1 — choose your molecule

# %%
SEAT = 0          # TODO: enter your seat number

MENU = [
 dict(id="water", name="water", smiles="O",
      bands="bend 1595, symmetric stretch 3657, antisymmetric stretch 3756",
      note="Three atoms, three vibrations. The simplest possible test."),
 dict(id="formaldehyde", name="formaldehyde", smiles="C=O",
      bands="CH2 wag 1167, CH2 rock 1249, CH2 scissor 1500, C=O stretch 1746, CH stretches 2782 and 2843",
      note="One mode is very weak. Ask which atoms move in it, and why that barely changes the dipole."),
 dict(id="methanol", name="methanol", smiles="CO",
      bands="C-O stretch 1033, CH3 deformations 1340-1477, CH stretches 2844-3000, O-H stretch 3681",
      note="The O-H stretch is the highest band and is strongly anharmonic."),
 dict(id="acetone", name="acetone", smiles="CC(C)=O",
      bands="C-C stretch 1216, CH3 deformations 1364-1435, C=O stretch 1731, CH stretches 2937-3019",
      note="The carbonyl band near 1730 is the most recognisable feature in organic IR."),
 dict(id="acetonitrile", name="acetonitrile", smiles="CC#N",
      bands="C-C stretch 918, CH3 deformations 1376-1448, C#N stretch 2267, CH stretches 2954-3009",
      note="The nitrile band sits in an otherwise empty region of the spectrum."),
 dict(id="chloroform", name="chloroform", smiles="ClC(Cl)Cl",
      bands="C-Cl stretches 668-774, CH bend 1220, CH stretch 3034",
      note="Heavy atoms give low frequencies. Compare your lowest band with water's lowest."),
 dict(id="benzene", name="benzene", smiles="c1ccccc1",
      bands="out-of-plane CH bend 673, in-plane CH bend 1037, C=C stretch 1486, CH stretches 3073",
      note="Highly symmetric, so many modes are infrared inactive. Count how many have "
           "non-negligible intensity and compare with 3N-6."),
 dict(id="ethanol", name="ethanol", smiles="CCO",
      bands="C-O stretch 1050, CH deformations 1250-1450, CH stretches 2900-3000, O-H stretch 3676",
      note="Compare the C-O and O-H positions with whoever has methanol."),
]

entry = MENU[SEAT % len(MENU)]
print(f"Molecule:   {entry['name']}   ({entry['smiles']})\n")
print(f"NOTE:       {entry['note']}\n")
print(f"EXPERIMENT: {entry['bands']}  (all in cm-1)")

# %% [markdown]
# ## Step 2 — optimise, then compute the Hessian
#
# The vibrational analysis is only valid at a stationary point. Frequencies
# computed at an unconverged geometry are meaningless, and the usual symptom is
# one or more small imaginary frequencies.

# %%
import time
from ase.optimize import LBFGS

molecule = attach(build(entry["smiles"]))
LBFGS(molecule, logfile=None).run(fmax=FMAX, steps=600)
print(f"optimised: largest force {np.abs(molecule.get_forces()).max():.4f} eV/A")

t0 = time.perf_counter()
H = molecule.calc.get_hessian(molecule)
numbers = molecule.get_atomic_numbers()
freqs, disp, n_rigid = normal_modes(H, numbers, molecule.get_positions())
intensities = ir_intensities(molecule, disp)
print(f"{len(numbers)} atoms -> {len(freqs)} vibrations, computed in "
      f"{time.perf_counter()-t0:.1f} s")

n_imaginary = int((freqs < -50).sum())
assert n_imaginary == 0, (
    f"{n_imaginary} imaginary frequencies: this is not a minimum. "
    "Re-run the optimisation with a smaller fmax or more steps.")
print("No imaginary frequencies: this is a genuine minimum.")

# %% [markdown]
# ## Step 3 — the band list

# %%
print(f"{'mode':>6}{'frequency / cm-1':>20}{'relative intensity':>22}")
print("-" * 52)
for k, (f, a) in enumerate(zip(freqs, intensities)):
    flag = "   weak" if a < 0.02 else ""
    print(f"{k+1:>6}{f:>20.0f}{a:>18.3f}   {'#' * int(round(a * 22))}{flag}")

# %% [markdown]
# ## Step 4 — the spectrum
#
# Infrared spectra are conventionally plotted with wavenumber decreasing to the
# right and with transmittance, rather than absorbance, on the vertical axis.
# Both conventions are used here so that the plot resembles a measured spectrum.

# %%
import matplotlib.pyplot as plt

x, y = spectrum(freqs, intensities)
fig, ax = plt.subplots(figsize=(8.2, 3.8))
ax.plot(x, 1 - y, color="#C41230", lw=1.6)
ax.set_xlim(4000, 400); ax.set_ylim(-0.05, 1.08)
ax.set_xlabel("wavenumber / cm$^{-1}$"); ax.set_ylabel("transmittance (arbitrary)")
ax.set_title(f"Predicted infrared spectrum of {entry['name']}")
for f, a in zip(freqs, intensities):
    if a > 0.10:
        ax.annotate(f"{f:.0f}", (f, 1 - a), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8, color="#6D6E71")
ax.grid(alpha=0.15)
fig.tight_layout(); plt.show()

# %% [markdown]
# ## Step 5 — assign the strongest band
#
# Identify which atoms move in the most intense mode. This is how a band is
# assigned: not by its position alone, but by the motion that produces it.

# %%
strongest = int(np.argmax(intensities))
d = disp[strongest]
d = d / np.linalg.norm(d, axis=1).max()
magnitude = np.linalg.norm(d, axis=1)

print(f"strongest band: {freqs[strongest]:.0f} cm-1\n")
print("atomic displacements, normalised:\n")
for i in np.argsort(-magnitude):
    print(f"  {chemical_symbols[int(numbers[i])]}{i:<3}{magnitude[i]:>7.2f}  "
          + "#" * int(round(magnitude[i] * 30)))

# %%
show(along_mode(molecule, d, amplitude=0.4), animate=True, loop="forward")

# %% [markdown]
# ## Step 6 — compare with experiment
#
# Compare each computed band with the experimental list printed in step 1.
# You should find that:
#
# - bending and skeletal modes, below about 1500 cm⁻¹, agree closely;
# - X–H stretching modes, above 2800 cm⁻¹, are overestimated by roughly 3 to 5
#   percent, and the stretches of multiple bonds such as C=O and C≡N by a
#   similar amount.
#
# That discrepancy is **not** a deficiency of the model. It is the harmonic
# approximation. A real bond is anharmonic: the true potential is shallower
# than a parabola at large displacement, which lowers the observed frequency.
# Quantum chemistry programs routinely apply an empirical scaling factor of
# about 0.96 to harmonic frequencies for this reason.

# %%
SCALE = 0.96
print(f"{'computed':>10}{'scaled by 0.96':>18}{'relative intensity':>24}")
print("-" * 52)
for f, a in zip(freqs, intensities):
    if a > 0.02:
        print(f"{f:>10.0f}  {f*SCALE:>16.0f}  {a:>20.3f}")

print(f"\nExperimental bands: {entry['bands']}")

# %% [markdown]
# ## Step 7 — your answer

# %%
ANSWER = ""     # TODO: Which band is strongest, what motion produces it, and how
                #       well do your positions match experiment? Two or three sentences.

assert len(ANSWER) > 40, "Please write a complete answer."
print(f"seat {SEAT}\t{entry['id']}\t{len(freqs)} modes\t"
      f"strongest {freqs[strongest]:.0f} cm-1\n{ANSWER}")

# %% [markdown]
# ## What this calculation does and does not include
#
# **Included:** the harmonic frequencies of an isolated molecule, and the
# intensity of each band under the dipole selection rule.
#
# **Not included:** anharmonicity, which lowers X–H stretches; overtones and
# combination bands, which appear weakly in real spectra; rotational fine
# structure; and any effect of solvent or of the sample being a liquid or solid,
# all of which broaden and shift bands. A real spectrum of a condensed sample is
# broader than the one plotted here, and hydrogen-bonded O–H stretches in
# particular are both shifted down and very much broader.

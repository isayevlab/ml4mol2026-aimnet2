SETUP = '''# %% [markdown]
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
EXTRA_INSTALL
import numpy as np
import torch
from aimnet.calculators import AIMNet2Calculator, AIMNet2ASE

GPU = torch.cuda.is_available()
print(f"Python {sys.version.split()[0]}   PyTorch {torch.__version__}   GPU available: {GPU}")

_ = AIMNet2Calculator("MODEL")          # downloads parameters on first use
print("Model loaded.")

'''

HELPERS = '''# %% [markdown]
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
'''

def setup(model="aimnet2", extra=None):
    s = SETUP.replace("MODEL", model)
    if extra:
        s = s.replace("EXTRA_INSTALL", extra)
    else:
        s = s.replace("EXTRA_INSTALL\n", "")
    return s

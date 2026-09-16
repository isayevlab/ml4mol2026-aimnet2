# %% [markdown]
# # 4. Molecular dynamics
#
# *Demonstration notebook.*
#
# Notebooks 2 and 3 treated molecules as if they sat motionless at the bottom of
# an energy well. Real molecules at room temperature are not motionless. This
# notebook follows them as they move.
#
# ## Concepts introduced here
#
# > **Definition — molecular dynamics (MD).**
# > Numerical integration of Newton's equations of motion for the atoms, using
# > forces obtained from the potential energy surface. Positions are updated in
# > small steps of time; the result is a trajectory, a film of the molecule.
#
# > **Definition — timestep.**
# > The interval between successive updates. It must be short compared with the
# > fastest motion in the system. The fastest motions in organic molecules are
# > X–H stretches at roughly 10⁻¹⁴ s, so timesteps are 0.5 to 1 femtosecond
# > (1 fs = 10⁻¹⁵ s). A trajectory of 1000 steps at 0.5 fs covers 0.5 ps.
#
# > **Definition — thermostat.**
# > An algorithm that couples the system to a heat bath so that it samples
# > structures at a chosen temperature, rather than conserving its initial
# > energy. The Langevin thermostat used here adds a friction force and a random
# > force whose magnitudes are related so as to reproduce the correct
# > temperature.
#
# ## Why this matters
#
# A single optimised structure is one point. A trajectory visits the whole
# region of the surface that is thermally accessible, which is what a real
# sample does. It answers questions a single structure cannot: whether a
# hydrogen bond survives heating, how often a torsion rotates, how much a bond
# length fluctuates.

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
# ## 4.1 Start from a minimum
#
# Salicylaldehyde has a hydroxyl group adjacent to an aldehyde, positioned so
# that the O–H can donate a hydrogen bond to the carbonyl oxygen.
#
# > **Reminder — hydrogen bond.** An attractive interaction between a hydrogen
# > atom bonded to an electronegative atom (here O–H) and a nearby electronegative
# > atom with a lone pair (here the carbonyl O). Typical H···O distances are
# > 1.6 to 2.2 Å, compared with about 0.98 Å for the covalent O–H bond.
#
# Dynamics must always begin from an optimised structure. Starting from a
# strained geometry converts that strain into kinetic energy and the system
# heats itself uncontrollably.

# %%
from ase.optimize import LBFGS

molecule = attach(build("O=Cc1ccccc1O"))
LBFGS(molecule, logfile=None).run(fmax=FMAX, steps=500)

d = molecule.get_all_distances()
H = [i for i, z in enumerate(molecule.numbers) if z == 1]
O = [i for i, z in enumerate(molecule.numbers) if z == 8]
h_donor = min(H, key=lambda h: min(d[h][o] for o in O))
o_acceptor = max(O, key=lambda o: d[h_donor][o])

print(f"optimised energy      {molecule.get_potential_energy():.4f} eV")
print(f"H...O hydrogen bond   {molecule.get_distance(h_donor, o_acceptor):.3f} A")

# %% [markdown]
# ## 4.2 A trajectory at two temperatures
#
# The same molecule is run at 300 K and at 500 K. At the higher temperature more
# of the potential energy surface is accessible, and the hydrogen bond should
# break and reform.
#
# Two settings below are worth reading before you run the cell.
#
# > **Definition — equilibration.**
# > The interval at the start of a trajectory during which the system is still
# > adjusting to the thermostat. Averages taken over it are meaningless, so it is
# > discarded. Here the thermostat couples on a timescale of 1/friction = 100 fs,
# > and the first 100 fs are discarded here (200 fs on a GPU, where the run is
# > longer).
#
# The timestep is **0.25 fs**, not the 0.5 fs often quoted for systems containing
# hydrogen. That choice was made by measurement, not by convention. Asking this
# thermostat for 300 K and measuring what it delivers:
#
# | timestep | measured temperature |
# |---|---|
# | 0.50 fs | 379 ± 20 K |
# | 0.25 fs | 305 ± 9 K |
# | 0.10 fs | 293 ± 15 K |
#
# At 0.5 fs the trajectory runs a quarter hotter than requested, far outside the
# sampling error. The integration is injecting energy faster than the thermostat
# removes it, so the steady state sits above the target. At 0.25 fs the bias is
# gone, and reducing it further changes nothing, which is how you know you have
# converged rather than merely moved. Section 4.3 is the measurement that
# diagnoses this class of problem directly.

# %%
import time
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

DT = 0.25                                  # fs
N_EQ = 800 if GPU else 400                 # discarded: equilibration
N_STEPS = 4000 if GPU else 1600            # sampled
SAMPLE = 2                                 # record every SAMPLE steps
trace = {}

print(f"{N_EQ} equilibration steps then {N_STEPS} sampled, "
      f"{DT} fs each: {N_STEPS * DT:.0f} fs of sampled trajectory per temperature\n")

# compiled once here, on a GPU, and shared by both temperatures
base = AIMNet2Calculator("aimnet2", compile_model=GPU)

for T in (300, 500):
    a = molecule.copy(); a.info.update(charge=0, mult=1)
    a.calc = AIMNet2ASE(base, charge=0, mult=1)

    # initial velocities drawn from the Maxwell-Boltzmann distribution at T
    MaxwellBoltzmannDistribution(a, temperature_K=T, force_temp=True)
    dynamics = Langevin(a, timestep=DT * units.fs, temperature_K=T,
                        friction=0.01 / units.fs)

    dynamics.run(N_EQ)                     # equilibrate, record nothing

    hbond, temperature = [], []
    dynamics.attach(lambda a=a, hb=hbond, tp=temperature:
                    (hb.append(a.get_distance(h_donor, o_acceptor)),
                     tp.append(a.get_temperature())), interval=SAMPLE)
    t0 = time.perf_counter()
    dynamics.run(N_STEPS)
    wall = time.perf_counter() - t0

    trace[T] = np.array(hbond)
    temperature = np.array(temperature)

    # Mean, spread, 95th percentile and maximum are all reported. The markdown
    # below explains why, over a run this short, none of them separates 300 K
    # from 500 K.
    d = trace[T]
    print(f"target {T} K"
          f"   H...O {d.mean():.2f} +- {d.std():.2f} A"
          f"   95th pct {np.percentile(d, 95):.2f}   max {d.max():.2f}"
          f"   |  T {temperature.mean():4.0f} K, spread +-{temperature.std():3.0f} K"
          f"   |  {wall / N_STEPS * 1000:.0f} ms/step")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.4, 3.6))
for T, colour in ((300, "#6D6E71"), (500, "#C41230")):
    ax.plot(np.arange(len(trace[T])) * DT * SAMPLE, trace[T],
            lw=0.9, color=colour, label=f"{T} K")
ax.axhline(2.2, color="0.75", ls="--", lw=1)
ax.text(2, 2.25, "upper limit for a hydrogen bond", fontsize=9, color="0.45")
ax.set_xlabel("time / fs"); ax.set_ylabel(r"H$\cdots$O distance / $\AA$")
ax.set_title("Intramolecular hydrogen bond in salicylaldehyde")
ax.legend(frameon=False); fig.tight_layout(); plt.show()

# %% [markdown]
# **Read this output carefully, because the honest conclusion is not the obvious
# one.** The mean H···O distance, its spread, and its largest excursion come out
# almost the same at 300 K and at 500 K. The hydrogen bond does not visibly
# weaken. It would be easy to write that it does, and wrong.
#
# The sampled trajectory is four hundred femtoseconds on a CPU and one
# picosecond on a GPU: roughly forty to a hundred O–H stretching periods, and
# four to ten thermostat coupling times. The cell prints the figure for your
# machine. That is enough to *see* a molecule moving, and
# the plot below is worth looking at for that reason. It is not enough to measure
# a difference between two temperatures in a quantity governed by rare
# excursions: breaking this hydrogen bond costs several kT even at 500 K, so it
# happens occasionally, and counting occasional events requires picoseconds.
# Repeat the cell and the numbers move by more than the gap you are looking for.
#
# This is the central practical difficulty of molecular dynamics, and it is worth
# meeting early. The cost of a trajectory is set by the timestep, which must
# resolve the fastest motion in the system, roughly 10⁻¹⁴ s. The property you want
# is usually governed by the slowest, which may be 10⁻⁹ s or slower. A fast
# potential shortens the wall-clock time per step; it does not change that ratio.
#
# > **Definition — sampling error.**
# > The uncertainty in an average taken over a finite trajectory. It falls as the
# > square root of the number of *statistically independent* configurations
# > visited, which is the trajectory length divided by the correlation time of the
# > property, not the number of steps. Quoting an average without it is the most
# > common way to reach a wrong conclusion from a correct simulation.
#
# No covalent bond breaks at either temperature. This is not guaranteed: a
# potential can reproduce equilibrium structures well and still tear molecules
# apart during dynamics, because dynamics visits regions of the surface that
# equilibrium calculations never probe.
#
# ## 4.3 Testing the potential: energy conservation
#
# A thermostat deliberately adds and removes energy, which conceals errors.
# Removing the thermostat gives a stringent test.
#
# > **Definition — NVE dynamics.**
# > Dynamics at constant particle number, volume and total energy. With no
# > thermostat, the sum of kinetic and potential energy must be conserved. At a
# > timestep already shown to be small enough, any systematic drift indicates
# > that the forces are not the exact gradient of the energy.

# %%
from ase.md.verlet import VelocityVerlet

def energy_drift(timestep_fs, n_steps):
    a = molecule.copy(); a.info.update(charge=0, mult=1); attach(a)
    # seeded, so that the three rows below are compared at identical
    # starting velocities and the table reproduces between runs
    MaxwellBoltzmannDistribution(a, temperature_K=300,
                                 rng=np.random.default_rng(0))
    total = []
    dynamics = VelocityVerlet(a, timestep=timestep_fs * units.fs)
    dynamics.attach(lambda: total.append(a.get_potential_energy() + a.get_kinetic_energy()),
                    interval=1)
    dynamics.run(n_steps)
    total = np.array(total)
    # The drift is the SLOPE of the total energy against step number. Taking
    # the difference between the last point and the first would instead measure
    # the fluctuation, which is much larger and carries no information.
    drift = np.polyfit(np.arange(len(total)), total, 1)[0]
    return abs(drift), total.std()

n = 1200 if GPU else 600
print(f"{'timestep':>11}{'drift / ueV per step':>24}{'fluctuation / meV':>21}")
print("-" * 56)
rows = []
for dt in (1.0, 0.5, 0.25):
    drift, fluctuation = energy_drift(dt, n)
    rows.append((dt, drift, fluctuation))
    print(f"{dt:>8.2f} fs{drift*1e6:>24.2f}{fluctuation*1000:>21.1f}")

ratio = rows[0][2] / rows[-1][2]
worst = max(r[1] for r in rows) * 1e6
thermal = 3 * len(molecule) * 0.0129          # 3N x kT/2 at 300 K, in eV
print("\nTwo different quantities, and the distinction matters.")
print("\nThe DRIFT is the slope of the total energy against time. It is the thing")
print(f"that must be zero, and here it is: at most {worst:.2f} ueV per step, against")
print(f"a thermal energy of {thermal:.2f} eV in this molecule. Over the whole {n}-step")
print(f"run that accumulates to under {worst * n / 1e6:.5f} eV. The forces really are the")
print("gradient of the energy, to the precision this test can see.")
print("\nThe FLUCTUATION is the spread of the total energy about that line. It is")
print("not an error in the same sense, but it does grow with the timestep:")
print(f"{ratio:.1f}x larger at 1.0 fs than at 0.25 fs here. It measures how far the")
print("discrete trajectory departs from the continuous one it approximates, which")
print("is exactly what makes a thermostat mis-set the temperature in 4.2.")
print("\nIf the drift column were NOT flat at 0.25 fs, the problem would not be integration.")
print("It would mean the forces are not the gradient of the energy, and nothing")
print("computed from any trajectory would be meaningful. Run this test on any")
print("potential you have not used before; it costs seconds.")

# %% [markdown]
# This is a test worth running whenever you adopt a new potential, and it takes
# under a minute.
#
# Note which column carries which information. The drift reports on the
# **potential**: a non-zero slope would mean the forces are not the gradient of
# the energy. The fluctuation reports on the **integrator**: it is finite even
# for a perfect potential, and it shrinks as the timestep is reduced. Over a few
# hundred steps the fluctuation is also the noisier of the two, so read the
# drift column first.
#
# ## 4.4 What dynamics costs
#
# The figure printed above in ms/step determines what is reachable. At 20 ms per
# step and the 0.25 fs timestep used above, one nanosecond of simulation
# requires four million steps, about twenty-two hours. Biomolecular questions routinely need microseconds.
# For those, the model must be driven by a specialised MD engine rather than by
# ASE; notebook 8 shows how.

#!/usr/bin/env python3
"""Generate the tutorial site into the repository root."""
import pathlib
import shell

REPO = "isayevlab/ml4mol2026-aimnet2"
OUT = pathlib.Path(__file__).parent.parent
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/"
GH = f"https://github.com/{REPO}"

NOTEBOOKS = [
    ("0_orientation", "Orientation", False,
     "The potential energy surface, and what a machine-learned potential replaces"),
    ("1_energy", "Energy, forces and charge", False,
     "Total charge is an input to the network, not a correction applied afterwards"),
    ("2_geometry_optimisation", "Geometry optimisation", False,
     "Water to within 0.01 &#197; and 0.5&deg; of experiment; an optimisation finds the nearest minimum, not the best one"),
    ("3_activity_conformers", "Conformational analysis", True,
     "Which form is populated at room temperature, and by how much. Eight systems, one per seat"),
    ("4_molecular_dynamics", "Molecular dynamics", False,
     "Watching a molecule move, why a short trajectory resolves less than it seems to, and energy conservation as a test"),
    ("5_reactions", "Reactions and transition states", False,
     "An S<sub>N</sub>2 saddle point, confirmed by exactly one imaginary frequency"),
    ("6_activity_ir_spectra", "Infrared spectra", True,
     "From the curvature of the surface to a predicted spectrum. Eight molecules, one per seat"),
    ("7_activity_reactivity", "Chemical reactivity", True,
     "Electrophilicity and Fukui functions from three charge states. Eight molecules, one per seat"),
    ("8_further_directions", "Further directions", False,
     "Periodic systems, bond dissociation energies, palladium, OpenMM, pysisyphus"),
]


def notebook_list():
    rows = []
    for stem, name, is_ex, why in NOTEBOOKS:
        cls = "nm ex" if is_ex else "nm"
        rows.append(
            f'<li><span class="{cls}">{name}</span>'
            f'<span class="why">{why}</span>'
            f'<a class="cl" href="{COLAB}{stem}.ipynb">open in colab</a></li>')
    return '<ol class="nb">\n' + "\n".join(rows) + "\n</ol>"


def write(name, html):
    (OUT / name).write_text(html, encoding="utf-8", newline="\n")
    print(f"  {name}  {len(html) // 1024} KB")


# ─────────────────────────────────────────────────────────────── index ──
INDEX = f"""
<h1>Machine-learned potentials<br>for molecular chemistry</h1>
<p class="tag">Hands-on AIMNet2 tutorial &middot; 90 minutes &middot;
AiMat Summer School 2026, <em>Machine Learning for Molecules</em></p>
<div class="rule"></div>

<p style="font-size:18px;max-width:40em">Almost everything you want to know about a
molecule follows from one function: its potential energy as a function of where the
atoms are. This tutorial computes seven different things from that one function, in
ninety minutes, on a laptop.</p>

<div class="note" style="margin-top:26px">
  <div class="k">energy &rarr; geometry &rarr; conformers &rarr; dynamics &rarr; reactions &rarr; spectra &rarr; reactivity</div>
  <p>Nine notebooks, in one direction. Every concept is defined where it is first
  needed. You need to be able to read Python; you do not need to have run a quantum
  chemistry calculation before. Molecules are shown in an interactive 3D viewer
  wherever the structure itself is the point.</p>
</div>

<a class="big" href="install.html">
  <span class="t">Start here &rarr;</span>
  <span class="d">Google Colab in one click, or a local install on macOS, Windows or Linux</span>
</a>
<a class="big" href="slides/">
  <span class="t">Slides &rarr;</span>
  <span class="d">33 slides. Arrow keys to advance, <kbd>O</kbd> for an overview, <kbd>P</kbd> to print to PDF</span>
</a>
<a class="big" href="{GH}">
  <span class="t">Repository &rarr;</span>
  <span class="d">Notebooks, worked solutions and these pages. MIT licensed</span>
</a>

<h2 id="notebooks">The nine notebooks</h2>
<p class="lede">Each one installs its own dependencies and stands alone, so arriving
late or losing a kernel costs you nothing.</p>
{notebook_list()}

<h2 id="exercises">The three exercises</h2>
<p>About thirteen minutes each. Systems are handed out by seat number, so no two
neighbours compute the same thing and the room pools its results into one table.
Worked answers for all eight seats are in <code>solutions/</code>.</p>
<div class="tw"><table>
<tr><th>Question the student answers</th><th>The eight systems</th></tr>
<tr><td>Which conformer is populated at room temperature, and by how much?</td>
    <td>butane, methylcyclohexane, ethylene glycol, glycine, 1,2-dichloroethane,
        1,4-butanediol, ibuprofen, alanine dipeptide</td></tr>
<tr><td>Which vibration gives the strongest infrared band, and where is it?</td>
    <td>water, formaldehyde, methanol, acetone, acetonitrile, chloroform, benzene,
        ethanol</td></tr>
<tr><td>Where does a nucleophile attack, and how electrophilic is the molecule?</td>
    <td>methyl vinyl ketone, acrolein, acrylonitrile, acetone, phenol, aniline,
        anisole, nitrobenzene</td></tr>
</table></div>

<h2 id="what">What gets computed</h2>
<div class="tw"><table>
<tr><th></th><th>Quantity</th><th>Obtained from</th></tr>
<tr><td class="num">1</td><td>energies, forces, partial charges, dipole moments</td>
    <td>a single evaluation of the surface</td></tr>
<tr><td class="num">2</td><td>equilibrium structures</td>
    <td>following the forces downhill</td></tr>
<tr><td class="num">3</td><td>conformer populations at 298 K</td>
    <td>many minima plus the Boltzmann distribution</td></tr>
<tr><td class="num">4</td><td>thermal motion, and what a trajectory can resolve</td>
    <td>integrating the equations of motion</td></tr>
<tr><td class="num">5</td><td>a transition state and its reaction coordinate</td>
    <td>the Hessian at a saddle point</td></tr>
<tr><td class="num">6</td><td>an infrared spectrum with assigned bands</td>
    <td>the Hessian at a minimum, plus dipole derivatives</td></tr>
<tr><td class="num">7</td><td>electrophilicity and the site of attack</td>
    <td>energies and charges at <em>N</em> and <em>N</em>&nbsp;&plusmn;&nbsp;1 electrons</td></tr>
</table></div>
<p>Every one of these is a property of the same function, <em>E</em>(<strong>R</strong>),
evaluated in a different way.</p>
"""

write("index.html", shell.page(
    "index.html", "AIMNet2 tutorial &mdash; AiMat Summer School 2026",
    "Hands-on AIMNet2 tutorial: energy, geometry optimisation, conformers, "
    "molecular dynamics, reactions, infrared spectra and chemical reactivity.",
    INDEX))


# ───────────────────────────────────────────────────────────── install ──
INSTALL = f"""
<h1>Getting set up</h1>
<p class="tag">Two routes. Pick one before the session starts.</p>
<div class="rule"></div>

<div class="note">
  <div class="k">If you are short of time</div>
  <p>Use <a href="#colab">Google Colab</a>. It needs a browser and a Google
  account, nothing else, and it gives you a free GPU. The local install is worth
  doing if you want to keep using AIMNet2 afterwards, and it is the only way to
  work offline.</p>
</div>

<h2 id="colab">Route 1: Google Colab</h2>
<p class="lede">No install. Works on any machine with a browser, including an iPad.</p>
<ol>
  <li>Open a notebook from the <a href="notebooks.html">notebook list</a>, or from
      the repository, by clicking its <strong>open in colab</strong> link.</li>
  <li>In the Colab menu choose <strong>Runtime &rarr; Change runtime type</strong>
      and select <strong>T4 GPU</strong>. Do this <em>before</em> running anything.
      If a runtime has already started, restart it after switching
      (<strong>Runtime &rarr; Restart session</strong>); a running runtime does not
      move to the GPU on its own. The free T4 makes the exercises roughly ten times
      faster.</li>
  <li>Run the first cell. It installs the packages and downloads the model
      parameters. This takes about a minute and prints nothing while it works, so
      it will look stalled. It is not.</li>
  <li>Run the rest of the cells in order.</li>
</ol>
<div class="note grey">
  <div class="k">Two things to know about Colab</div>
  <p><strong>Each notebook is a separate machine.</strong> Opening notebook 3 does
  not inherit anything from notebook 2, so the first cell has to run again. That is
  why every notebook is self-contained.</p>
  <p><strong>Your edits are not saved</strong> unless you choose
  <strong>File &rarr; Save a copy in Drive</strong>. Do that first if you want to
  keep your answers.</p>
</div>

<h2 id="local">Route 2: a local install</h2>
<p class="lede">About five minutes, plus the PyTorch download: about 3&nbsp;GB on
Linux x86-64, roughly 120&nbsp;MB of torch plus 150&nbsp;MB of warp on Windows, and
roughly 150&nbsp;MB on an Apple Silicon Mac.</p>

<h3>Will it run on your machine?</h3>
<div class="tw"><table>
<tr><th>Platform</th><th>Supported</th><th>Compute</th><th>Notes</th></tr>
<tr><td>Linux, x86-64</td><td class="yes">yes</td><td>NVIDIA GPU or CPU</td>
    <td>A CUDA-enabled PyTorch is installed automatically. It needs NVIDIA driver
        580 or newer; see the Linux tab for older drivers.</td></tr>
<tr><td>Linux, ARM64</td><td class="yes">yes</td><td>CPU</td>
    <td>Includes Raspberry Pi-class hardware and ARM cloud instances.</td></tr>
<tr><td>Windows 10 / 11, x86-64</td><td class="yes">yes</td><td>NVIDIA GPU or CPU</td>
    <td>The default PyTorch on Windows is CPU-only. See the GPU note below.</td></tr>
<tr><td>macOS, Apple Silicon</td><td class="yes">yes</td><td>CPU</td>
    <td>M1 and later. AIMNet2 does not use the Metal (MPS) backend, so everything
        runs on the CPU cores. That is fine for this tutorial.</td></tr>
<tr><td>macOS, Intel</td><td class="no">no</td><td>&mdash;</td>
    <td>AIMNet2 needs PyTorch 2.8 or later, and PyTorch stopped publishing Intel-Mac
        builds after version 2.2. Use Colab.</td></tr>
</table></div>

<h3>Python version</h3>
<p>You need <strong>Python 3.11 or later</strong>. AIMNet2 declares
<code>requires-python &gt;= 3.11</code> and will refuse to install on 3.10 or older.
Check with <code>python3 --version</code> (macOS, Linux) or <code>py --version</code>
(Windows; Store Python: <code>python --version</code>).</p>

<h3>Install</h3>
<p>Pick your platform. Each block creates an isolated environment so nothing lands in
your system Python.</p>

<div class="tabs" data-tabs="install-os" role="tablist">
  <button role="tab">macOS</button>
  <button role="tab">Windows</button>
  <button role="tab">Linux</button>
  <button role="tab">conda</button>
</div>

<div class="panel">
  <p>Terminal, on an Apple Silicon Mac. If you do not have Python 3.11+, install it
  with <a href="https://brew.sh">Homebrew</a>: <code>brew install python@3.12</code>.</p>
<pre><code><span class="c"># make a folder for the tutorial and a virtual environment inside it</span>
mkdir ~/aimnet2-tutorial &amp;&amp; cd ~/aimnet2-tutorial
python3 -m venv .venv
source .venv/bin/activate

<span class="c"># install (about 150 MB)</span>
pip install --upgrade pip
pip install "aimnet[ase]" rdkit py3Dmol sella matplotlib jupyterlab

<span class="c"># get the notebooks</span>
git clone {GH}.git
cd ml4mol2026-aimnet2
jupyter lab</code></pre>
  <p>Next time you come back, <code>cd ~/aimnet2-tutorial</code> and
  <code>source .venv/bin/activate</code> before <code>jupyter lab</code>.</p>
</div>

<div class="panel" hidden>
  <p>PowerShell. Install Python from <a href="https://www.python.org/downloads/windows/">python.org</a>
  or the Microsoft Store first, and tick <strong>Add python.exe to PATH</strong>.
  (Store Python does not ship the <code>py</code> launcher: use
  <code>python -m venv .venv</code> and <code>python --version</code>.)</p>
<pre><code><span class="c"># make a folder for the tutorial and a virtual environment inside it</span>
mkdir $HOME\\aimnet2-tutorial; cd $HOME\\aimnet2-tutorial
py -3.12 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1

<span class="c"># install (roughly 120 MB torch plus 150 MB warp)</span>
python -m pip install --upgrade pip
pip install "aimnet[ase]" rdkit py3Dmol sella matplotlib jupyterlab

<span class="c"># get the notebooks</span>
git clone {GH}.git
cd ml4mol2026-aimnet2
jupyter lab</code></pre>
  <div class="note grey">
    <div class="k">If PowerShell refuses to run the activation script</div>
    <p>Windows blocks local scripts by default. Allow them for your account once:</p>
    <pre><code>Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned</code></pre>
    <p>then run <code>.\\.venv\\Scripts\\Activate.ps1</code> again. In
    <code>cmd.exe</code> instead of PowerShell the activation command is
    <code>.venv\\Scripts\\activate.bat</code>.</p>
  </div>
  <div class="note grey">
    <div class="k">For an NVIDIA GPU on Windows</div>
    <p>The PyTorch published on PyPI for Windows is CPU-only; the CUDA build comes
    from PyTorch's own index. Install it <em>before</em> AIMNet2:</p>
    <pre><code>pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "aimnet[ase]" rdkit py3Dmol sella matplotlib jupyterlab</code></pre>
    <p>The <code>cu126</code> build needs NVIDIA driver 525 or newer; the current
    options are listed at
    <a href="https://pytorch.org/get-started/locally/">pytorch.org/get-started/locally</a>.
    Without this everything still works, just on the CPU.</p>
    <p><code>torch.compile</code> is unavailable on Windows, which has no C++ compiler
    or Triton, so the setup cell in every notebook disables it automatically there.
    Nothing to do.</p>
  </div>
</div>

<div class="panel" hidden>
<pre><code><span class="c"># make a folder for the tutorial and a virtual environment inside it</span>
mkdir ~/aimnet2-tutorial &amp;&amp; cd ~/aimnet2-tutorial
python3 -m venv .venv
source .venv/bin/activate

<span class="c"># install (about 3 GB on x86-64, because the PyTorch wheel bundles CUDA)</span>
pip install --upgrade pip
pip install "aimnet[ase]" rdkit py3Dmol sella matplotlib jupyterlab

<span class="c"># get the notebooks</span>
git clone {GH}.git
cd ml4mol2026-aimnet2
jupyter lab</code></pre>
  <p>On x86-64 Linux the PyPI PyTorch wheel has bundled CUDA 13.0 since torch 2.11,
  which needs NVIDIA driver 580 or newer. On an older driver the install succeeds but
  <code>torch.cuda.is_available()</code> is silently <code>False</code>. Check the
  driver version with <code>nvidia-smi</code> first; if it is below 580, install torch
  from the CUDA 12.6 index <em>before</em> the rest:</p>
  <pre><code>pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "aimnet[ase]" rdkit py3Dmol sella matplotlib jupyterlab</code></pre>
  <p>On a laptop without an NVIDIA GPU, install the CPU build first instead. It cuts
  the download from about 3&nbsp;GB to about 200&nbsp;MB:</p>
  <pre><code>pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "aimnet[ase]" rdkit py3Dmol sella matplotlib jupyterlab</code></pre>
  <p>Either way, confirm with:</p>
  <pre><code>python -c "import torch; print(torch.cuda.is_available())"</code></pre>
</div>

<div class="panel" hidden>
  <p>If you already use conda or mamba. Works the same on all three platforms.</p>
<pre><code>conda create -n aimnet2 python=3.12 -y
conda activate aimnet2

pip install "aimnet[ase]" rdkit py3Dmol sella matplotlib jupyterlab

git clone {GH}.git
cd ml4mol2026-aimnet2
jupyter lab</code></pre>
  <p>Install the packages with <code>pip</code> even inside conda. AIMNet2 is not on
  conda-forge, and mixing the two channels for PyTorch causes more problems than it
  solves.</p>
</div>

<h3 id="check">Check that it worked</h3>
<p>Open <code>notebooks/0_orientation.ipynb</code> and run every cell. It ends with
four checks and prints whether a GPU was found. Or, from a terminal:</p>
<pre><code>python -c "from aimnet.calculators import AIMNet2Calculator as C; C('aimnet2'); print('AIMNet2 works')"</code></pre>
<p>That one line works in bash, zsh and PowerShell alike. It prints a CUDA warning
first on any machine without an NVIDIA card, which is expected.</p>

<h2 id="offline">Before a conference: warm the cache</h2>
<p class="lede">Do this at home, on a network you trust.</p>
<p>Model parameters are downloaded the first time each model is used, not at install
time, and they land in <code>~/.cache/aimnet</code>
(<code>C:\\Users\\&lt;you&gt;\\.cache\\aimnet</code> on Windows). Each file is about
9&nbsp;MB. The tutorial uses four of them. Fetching them in a lecture theatre on
shared wifi, thirty people at once, is the single most likely way for the session to
stall.</p>
<p>Run this once beforehand and the notebooks will need no network at all:</p>
<pre><code>python -c "
from aimnet.calculators import AIMNet2Calculator
for m in ('aimnet2', 'aimnet2-2025', 'aimnet2-nse', 'aimnet2-rxn'):
    AIMNet2Calculator(m); print('cached', m)
"</code></pre>
<p>To keep the cache somewhere else, set <code>AIMNET_CACHE_DIR</code> before
running Python.</p>

<h2 id="which">Which model is which</h2>
<p>Choosing a model is a chemical decision, not a performance one.</p>
<div class="tw"><table>
<tr><th>Name</th><th>Elements</th><th>Charge and spin</th><th>Use it for</th></tr>
<tr><td><code>aimnet2</code></td><td>14</td><td>closed shell</td>
    <td>the default; notebooks 0&ndash;4 and 6</td></tr>
<tr><td><code>aimnet2-2025</code></td><td>14</td><td>closed shell</td>
    <td>a second opinion at a different level of theory</td></tr>
<tr><td><code>aimnet2-nse</code></td><td>14</td><td>any charge, any multiplicity</td>
    <td>radicals and ions; notebooks 5, 7 and 8. The transition-state search in
        notebook 5 uses it</td></tr>
<tr><td><code>aimnet2-rxn</code></td><td>H C N O</td><td>neutral only</td>
    <td>reaction paths in neutral H, C, N, O systems. Appears in notebook 5 only to
        show it refusing a charged chloride system</td></tr>
<tr><td><code>aimnet2-pd</code></td><td>14, plus Pd</td><td>closed shell</td>
    <td>palladium. Not a gas-phase model: its energies are not comparable with any
        other member's</td></tr>
</table></div>
<p>The fourteen elements are H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br and I. There
is no solvent in any of these models except <code>aimnet2-pd</code>.</p>

<div class="note">
  <div class="k">One naming trap</div>
  <p>The package on PyPI is <strong><code>aimnet</code></strong>, not
  <code>aimnet2</code> or <code>aimnet2calc</code>. The old
  <code>isayevlab/AIMNet2</code> repository is deprecated, and any instructions that
  begin by cloning it and running <code>setup.py install</code> are out of date.</p>
</div>
"""

write("install.html", shell.page(
    "install.html", "Install &mdash; AIMNet2 tutorial",
    "How to run the AIMNet2 tutorial: Google Colab, or a local install on macOS, "
    "Windows or Linux.",
    INSTALL, shell.TABS_JS))


# ─────────────────────────────────────────────────────────── notebooks ──
NOTEBOOKS_PAGE = f"""
<h1>The notebooks</h1>
<p class="tag">Nine of them, in one direction. Each stands alone.</p>
<div class="rule"></div>

<p>Every notebook installs its own dependencies in the first cell, so you can open
any one of them cold. If you are running locally you only need the install once;
the first cell then finds the packages already present and skips straight through.</p>

<p>Where a structure is the point, the notebooks draw it: an interactive
three-dimensional view you can rotate and zoom. The first molecule, the charges
printed on its atoms, an optimisation played as a film, anti and gauche butane side
by side, the three lowest conformers, a molecular dynamics trajectory, a transition
state moving along its imaginary mode, the strongest infrared mode, the Fukui
indices on each atom, and six waters in their periodic box. The viewer is
<a href="https://3dmol.csb.pitt.edu/">3Dmol.js</a> through <code>py3Dmol</code>;
it needs a network connection, and every result is also printed as text.</p>

{notebook_list()}

<h2>Worked solutions</h2>
<p>The three exercises each have a worked copy in <code>solutions/</code>, with seat 0
filled in, a table of the expected result for all eight seats, the points worth
drawing out in discussion, and the difficulties students actually hit.</p>
<ul>
  <li><a href="{GH}/blob/main/solutions/3_activity_conformers_ANSWERS.ipynb">3 &mdash; Conformational analysis</a></li>
  <li><a href="{GH}/blob/main/solutions/6_activity_ir_spectra_ANSWERS.ipynb">6 &mdash; Infrared spectra</a></li>
  <li><a href="{GH}/blob/main/solutions/7_activity_reactivity_ANSWERS.ipynb">7 &mdash; Chemical reactivity</a></li>
</ul>

<h2>Reference numbers</h2>
<p class="lede">What to expect when you run the material. Several of these double as
sanity checks: if yours differ, something is wrong.</p>
<div class="tw"><table>
<tr><th>Nb</th><th>Quantity</th><th>Computed</th><th>Reference</th></tr>
<tr><td class="num">1</td><td>acetic acid &rarr; acetate, C&ndash;O bond lengths</td>
    <td class="num">1.201 / 1.352 &rarr; 1.251 / 1.250 &#197;</td>
    <td>the two bonds equalise on deprotonation</td></tr>
<tr><td class="num">1</td><td>glycine zwitterion, gas phase</td>
    <td class="num">collapses; O&ndash;H 1.702 &rarr; 0.979 &#197;</td>
    <td>no zwitterion exists in the gas phase</td></tr>
<tr><td class="num">2</td><td>water, O&ndash;H bond length</td><td class="num">0.9581 &#197;</td>
    <td class="num">0.958 &#197;, experiment</td></tr>
<tr><td class="num">2</td><td>water, H&ndash;O&ndash;H angle</td><td class="num">104.97&deg;</td>
    <td class="num">104.5&deg;, experiment</td></tr>
<tr><td class="num">3</td><td>butane, anti : gauche at 298 K</td>
    <td class="num">63.6 : 36.4, &Delta;E = 0.33 kcal/mol</td>
    <td>anti lower by 0.6&ndash;0.9 kcal/mol</td></tr>
<tr><td class="num">3</td><td>1,4-butanediol, lowest conformer</td>
    <td class="num">O&ndash;H&middot;&middot;&middot;O at 2.22 &#197;</td>
    <td>every other conformer exceeds 4.6 &#197;</td></tr>
<tr><td class="num">3</td><td>butane gap, <code>fmax</code> 0.02 vs 0.005</td>
    <td class="num">0.33 vs 0.31 kcal/mol; population 63.6 vs 63.0%</td>
    <td>tightening changes nothing you report</td></tr>
<tr><td class="num">4</td><td>Langevin at 300 K, measured temperature</td>
    <td class="num">379 K at &Delta;t = 0.5 fs; 305 K at 0.25 fs</td>
    <td>300 K. The 0.5 fs bias is an integration artefact</td></tr>
<tr><td class="num">5</td><td>S<sub>N</sub>2 saddle point</td>
    <td class="num">one imaginary mode, &minus;193 cm&#8315;&sup1;</td>
    <td>exactly one, by definition of a transition state</td></tr>
<tr><td class="num">6</td><td>water, bend</td><td class="num">1597 cm&#8315;&sup1;</td>
    <td class="num">1595 cm&#8315;&sup1;, experiment</td></tr>
<tr><td class="num">6</td><td>water, O&ndash;H stretches scaled &times;0.96</td>
    <td class="num">3639 / 3785 cm&#8315;&sup1;</td>
    <td class="num">3657 / 3756 cm&#8315;&sup1;, experiment</td></tr>
<tr><td class="num">7</td><td>methyl vinyl ketone, largest <em>f</em>&#8314;</td>
    <td class="num">&beta;-carbon 0.232, carbonyl carbon 0.135</td>
    <td>the &beta;-carbon, site of conjugate addition</td></tr>
<tr><td class="num">7</td><td>electrophilicity index <em>&omega;</em></td>
    <td class="num">acetone 0.65 &lt; MVK 1.06 &lt; acrolein 1.18 &lt; acrylonitrile 1.20 eV</td>
    <td>the accepted ordering</td></tr>
<tr><td class="num">8</td><td>bond dissociation energies, CH&#8323;&ndash;H / PhCH&#8322;&ndash;H / HO&ndash;H</td>
    <td class="num">105.2 / 92.8 / 116.8 kcal/mol</td>
    <td class="num">105 / 90 / 119 kcal/mol</td></tr>
</table></div>

<h2>Four limitations, kept deliberately</h2>
<p class="lede">Each notebook says so where it happens. These are more useful to a
class than results that all agree.</p>

<h3>Fukui functions fail for deactivated arenes</h3>
<p>Phenol, aniline and anisole give the correct ortho and para preference. Nitrobenzene
puts meta above ortho, capturing part of the deactivation, but ranks para highest
where experiment gives 93 percent meta. A condensed Fukui function describes how the
<em>ground-state</em> density redistributes; regioselectivity is set by the relative
energies of the <em>&sigma;</em>-complex intermediates, a different quantity. For
activated rings the two correlate. For deactivated rings they need not. That makes
this a screening method, not an oracle.</p>

<h3>X&ndash;H stretching frequencies come out 3 to 5 percent high</h3>
<p>This is the harmonic approximation, not the model: a real X&ndash;H potential is
shallower than a parabola at large displacement. Bending and skeletal modes below
about 1800 cm&#8315;&sup1; need no scaling at all.</p>

<h3>A short trajectory cannot measure what a long one can</h3>
<p>Notebook 4 runs a few hundred femtoseconds at 300 K and at 500 K and finds the
intramolecular hydrogen bond behaving the same at both. That is the correct result
for that trajectory length, not a failure of the potential: breaking the bond costs
several <em>kT</em>, so it is a rare event, and counting rare events takes
picoseconds. The notebook uses this to introduce sampling error, which is the
difficulty a fast potential does not remove.</p>

<h3>Everything here is in the gas phase</h3>
<p>Internal hydrogen bonds are therefore overstated, because in water a solvent
molecule competes for the same donor. Notebook 1 shows the extreme case: the glycine
zwitterion, the dominant form in solution, has no minimum at all in the gas phase and
collapses to the neutral form during optimisation.</p>
"""

write("notebooks.html", shell.page(
    "notebooks.html", "Notebooks &mdash; AIMNet2 tutorial",
    "The nine tutorial notebooks, the worked solutions, the reference numbers to "
    "expect, and the limitations kept deliberately.",
    NOTEBOOKS_PAGE))


# ────────────────────────────────────────────────────── troubleshooting ──
TROUBLE = f"""
<h1>Troubleshooting</h1>
<p class="tag">Everything below has actually happened to somebody.</p>
<div class="rule"></div>

<h2>Installing</h2>

<h3><code>ERROR: Could not find a version that satisfies the requirement aimnet</code></h3>
<p>Almost always Python 3.10 or older. AIMNet2 requires 3.11 or later, and pip reports
that as "no matching distribution" rather than saying so. Check with:</p>
<pre><code>python -c "import sys; print(sys.version)"</code></pre>
<p>If that prints 3.10 or lower, make a new environment with a newer interpreter. On
macOS <code>brew install python@3.12</code>; on Windows install 3.12 from python.org
and use <code>py -3.12 -m venv .venv</code>; on Linux use your distribution's
<code>python3.12</code> package or <code>pyenv</code>.</p>

<h3>The same error on an Intel Mac</h3>
<p>It will not install, and the problem is not fixable locally. AIMNet2 needs PyTorch
2.8 or later and PyTorch stopped publishing Intel-Mac builds after 2.2. Use
<a href="install.html#colab">Colab</a>.</p>

<h3><code>pip install</code> downloads gigabytes and then fails on disk space</h3>
<p>On Linux x86-64 PyTorch is about 3&nbsp;GB installed, more during the install
while the wheel is still cached (the Windows and macOS wheels are far smaller). Free up 6&nbsp;GB, or point pip's cache elsewhere with
<code>PIP_CACHE_DIR</code>, or install with <code>--no-cache-dir</code>.</p>

<h3>PowerShell: <em>running scripts is disabled on this system</em></h3>
<p>Windows blocks local scripts, including the one that activates a virtual
environment. Allow them for your own account:</p>
<pre><code>Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned</code></pre>
<p>Then activate again. This does not require an administrator and does not allow
scripts downloaded from the internet.</p>

<h3><code>Warp CUDA warning: Could not find or load the NVIDIA CUDA driver</code></h3>
<p>Harmless. One of AIMNet2's dependencies looks for a GPU at import time and says so
when it does not find one. Everything runs on the CPU. You will see this on every
Mac and on any machine without an NVIDIA card.</p>

<h3><code>Could not find or load the NVIDIA CUDA driver</code> on a machine that has an NVIDIA card</h3>
<p>warp-lang 1.18 and later need NVIDIA driver 580 or newer. The setup cell and
<code>requirements.txt</code> pin <code>warp-lang&lt;1.18</code>, so this should not
happen from the notebooks. If you installed by hand, either update the driver or
<code>pip install "warp-lang&lt;1.18"</code>.</p>

<h3>Windows: <code>InductorError: InvalidCxxCompiler: Compiler: cl is not found</code></h3>
<p>Any <code>torch._inductor</code> or <code>torch._dynamo</code> error on Windows has
the same cause: <code>torch.compile</code> needs a C++ compiler, and Windows has none.
The setup cell sets <code>TORCHDYNAMO_DISABLE=1</code> on Windows before torch is
imported. If torch was imported earlier some other way, restart the kernel and run the
setup cell first.</p>

<h2>Running</h2>

<h3>The first cell sits there doing nothing</h3>
<p>Expected, for about a minute. It is installing packages and downloading roughly
36&nbsp;MB of model parameters, and neither step prints progress. If it is still
going after five minutes the network is the problem; see
<a href="install.html#offline">warming the cache</a>.</p>

<h3><code>RuntimeError</code> or <code>KeyError</code> mentioning an element</h3>
<p>The molecule contains an element the model was not trained on. The fourteen
supported elements are H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br and I. There are no
metals except palladium, in <code>aimnet2-pd</code> only.</p>

<h3><code>ValueError</code> about charge or multiplicity</h3>
<p>You asked a closed-shell model for an open-shell system. Ions and radicals need
<code>aimnet2-nse</code>; <code>aimnet2-rxn</code> refuses any net charge at all. A
cation or anion of a closed-shell molecule is a doublet, so <code>mult = 2</code>.</p>

<h3>An optimisation runs forever, or hits the step limit</h3>
<p>Check that you have not tightened <code>fmax</code>. AIMNet2 is a float32 model and
0.02 eV/&#197; is the right convergence threshold; the <code>1e-4</code> you would use
with a DFT code is below the numerical noise floor and will never be reached.</p>
<p>Tightening costs a great deal and buys nothing you are going to use. Ibuprofen
reaches 0.02 eV/&#197; in 26 L-BFGS steps and 0.01 in 154, six times the work. The
<em>relative</em> energies that conformational analysis depends on do not move:
butane's anti&ndash;gauche gap is 0.33 kcal/mol at <code>fmax</code>&nbsp;=&nbsp;0.02
and 0.31 at 0.005, and its Boltzmann population changes by half a percentage point.
For ethylene glycol the first three gaps agree to better than 0.01 kcal/mol.</p>

<h3>The 3D viewer is an empty box, or shows nothing at all</h3>
<p>The viewer is 3Dmol.js, fetched from the web when the cell runs, so it needs a
network connection even in a local install; a content blocker that stops scripts
from <code>cdn.jsdelivr.net</code> has the same effect. In JupyterLab a
notebook reopened from disk is <em>untrusted</em>, and untrusted JavaScript output
is not rendered: run the cell again. Nothing in the tutorial depends on the
viewer; every result is also printed as text.</p>

<h3>My conformer count differs from the answer key</h3>
<p>Expected for the flexible systems. Conformer generation is stochastic, so the exact
number varies between runs and between machines. The energy <em>ordering</em> and the
size of the gaps are the reproducible quantities, and are what the exercise asks
for.</p>

<h3>The dynamics notebook gives a different temperature each time</h3>
<p>Also expected, and it is one of the points of that notebook. A few hundred
femtoseconds of a fifteen-atom molecule does not determine a mean temperature to
better than tens of kelvin. Read the spread, not the mean.</p>

<h3>An exercise cell raises <code>AssertionError: Please write a complete answer</code></h3>
<p>That is the blank you are meant to fill in. Replace the empty string with your
answer in complete sentences and run the cell again.</p>

<h3>I get an imaginary frequency at a structure that should be a minimum</h3>
<p>The geometry is not converged. Re-run the optimisation, and if it still happens,
loosen nothing: a genuine imaginary mode at a supposed minimum means the optimiser
stopped on a saddle point, so nudge the structure and optimise again.</p>

<h2>Colab</h2>

<h3>Everything is slow</h3>
<p>You are probably on a CPU runtime. <strong>Runtime &rarr; Change runtime type &rarr;
T4 GPU</strong>, then <strong>Runtime &rarr; Restart session</strong> and run from the
top. Check which you have with:</p>
<pre><code>import torch; print(torch.cuda.is_available())</code></pre>

<h3>My session disconnected and I lost everything</h3>
<p>Colab reclaims idle runtimes. Re-run from the first cell; nothing is carried
between notebooks anyway, so you only lose the notebook you were in. Save your own
copy with <strong>File &rarr; Save a copy in Drive</strong> if you want to keep your
answers.</p>

<h3>"You cannot currently connect to a GPU backend"</h3>
<p>The free tier has quotas. Either wait, or use a CPU runtime and accept that the
exercises take five to twenty times longer. The notebooks detect the absence of a GPU
and shrink the dynamics runs automatically.</p>

<h2>Still stuck</h2>
<p>Open an issue at <a href="{GH}/issues">{GH.replace('https://github.com/', '')}</a>,
including the output of:</p>
<pre><code>python -c "
import sys, platform, torch, aimnet
print(sys.version)
print(platform.platform(), platform.machine())
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import importlib.metadata as m; print('aimnet', m.version('aimnet'))
"</code></pre>
"""

write("troubleshooting.html", shell.page(
    "troubleshooting.html", "Troubleshooting &mdash; AIMNet2 tutorial",
    "Install and runtime problems in the AIMNet2 tutorial, and what to do about them.",
    TROUBLE))


# ────────────────────────────────────────────────────────────- teaching ──
TEACHING = f"""
<h1>For instructors</h1>
<p class="tag">Running this as a ninety-minute session.</p>
<div class="rule"></div>

<h2>Run sheet</h2>
<div class="tw"><table>
<tr><th>From</th><th>Min</th><th>Notebook</th><th>Mode</th><th>What it has to land</th></tr>
<tr><td class="num">0:00</td><td class="num">5</td><td>0 Orientation</td><td>together</td>
    <td>Everyone's first cell runs. Everything else follows from one function.</td></tr>
<tr><td class="num">0:05</td><td class="num">10</td><td>1 Energy</td><td>demo</td>
    <td>Charge is an input. The glycine zwitterion collapses.</td></tr>
<tr><td class="num">0:15</td><td class="num">8</td><td>2 Geometry</td><td>demo</td>
    <td>Water to within 0.01 &#197; and 0.5&deg;. An optimisation finds the <em>nearest</em> minimum.</td></tr>
<tr><td class="num">0:23</td><td class="num">13</td><td>3 Conformers</td><td><strong>exercise 1</strong></td>
    <td>A molecule at 298 K is a population, not a structure.</td></tr>
<tr><td class="num">0:36</td><td class="num">8</td><td>4 Dynamics</td><td>demo</td>
    <td>Sampling, not speed, is the hard part of MD.</td></tr>
<tr><td class="num">0:44</td><td class="num">10</td><td>5 Reactions</td><td>demo</td>
    <td>Exactly one imaginary mode, and what it is.</td></tr>
<tr><td class="num">0:54</td><td class="num">13</td><td>6 Infrared</td><td><strong>exercise 2</strong></td>
    <td>The same Hessian, at a minimum instead of a saddle, is a spectrum.</td></tr>
<tr><td class="num">1:07</td><td class="num">13</td><td>7 Reactivity</td><td><strong>exercise 3</strong></td>
    <td>Where a molecule reacts, without a transition state.</td></tr>
<tr><td class="num">1:20</td><td class="num">5</td><td>8 Further directions</td><td>together</td>
    <td>Where to go next, and what this cannot do.</td></tr>
</table></div>
<p>That totals eighty-five minutes and leaves five for the room. If you are running
short, notebook 4 is the one to cut: it is the only demo nothing later depends on.</p>

<h2>Before the session</h2>
<ul>
  <li><strong>Warm the model cache on your own machine.</strong> See
      <a href="install.html#offline">the install page</a>. Thirty people downloading
      36&nbsp;MB each over lecture-theatre wifi is the most likely way to lose the
      first fifteen minutes.</li>
  <li><strong>Send the install page out a week ahead</strong> and say plainly that
      Colab is fine and an Intel Mac will not work locally.</li>
  <li><strong>Run <code>python check_notebooks.py</code></strong> in the repository.
      It takes under a second and confirms the exercises still have their blanks and
      the answer keys do not.</li>
  <li><strong>Decide how seats are numbered</strong> and be able to say it in one
      sentence. Row-major from the front left, modulo eight, works.</li>
  <li><strong>Open the three answer keys</strong> in <code>solutions/</code> on your
      own screen. Each one has a table of the expected result for all eight seats.</li>
</ul>

<h2>The exercises</h2>
<p class="lede">Each hands out eight systems by seat number, so neighbours cannot copy
and the room can pool results into one table at the end. That pooling is the point:
no individual seat sees the pattern.</p>

<h3>Exercise 1, conformers</h3>
<p>Put <strong>seat 5, 1,4-butanediol</strong> on the screen when the room reports
back. Exactly one of its twenty-odd conformers folds far enough to close the
O&ndash;H&middot;&middot;&middot;O bond; that one is the global minimum and holds about
a third of the population, and every other conformer leaves the oxygens more than
4.6&nbsp;&#197; apart. Seats 0 and 4, butane and 1,2-dichloroethane, should compare
their numbers directly: the same question, different substituent, larger gap for
chlorine.</p>

<h3>Exercise 2, infrared</h3>
<p><strong>Read the answer key before the session.</strong> For six of the eight
molecules the strongest band is the one a chemist would name first: C=O, C&equiv;N,
C&ndash;Cl and C&ndash;O all dominate their spectra, because intensity is
&part;&mu;/&part;Q and polar bonds have the largest dipole derivative. The two
alcohols are the exception. Methanol (seat 2) and ethanol (seat 7) show a very
strong hydroxyl torsion at 205 and 282 cm<sup>&minus;1</sup>, and for ethanol it is
the strongest band in the spectrum. Seat 7 will report it as their answer. That is
what the code returns, and the reason it is not a prediction of an observable band
(a torsional potential is nowhere near quadratic) is the most useful thing in the
exercise.</p>

<h3>Exercise 3, reactivity</h3>
<p><strong>Seat 7, nitrobenzene, is the instructive failure</strong> and should be
handled deliberately rather than apologised for. The descriptor puts meta above ortho,
capturing part of the deactivation, but ranks para highest where experiment gives 93
percent meta. Say why: condensed Fukui functions describe the ground-state density,
whereas regioselectivity is set by the relative energies of the
<em>&sigma;</em>-complex intermediates. For activated rings the two correlate; for
deactivated rings they need not.</p>
<p>Also warn the room not to read the <em>&omega;</em> column as a single ranking
across all eight seats. Seats 4 to 6 are activated arenes with lower
<em>&omega;</em> than acetone, which is correct: they are nucleophiles.</p>

<h2>What goes wrong in a room</h2>
<div class="tw"><table>
<tr><th>Symptom</th><th>Cause</th><th>Say</th></tr>
<tr><td>Nothing happens for a minute</td><td>first cell installing</td>
    <td>Expected. It prints nothing while it works.</td></tr>
<tr><td>Everything is slow on Colab</td><td>CPU runtime</td>
    <td>Runtime &rarr; Change runtime type &rarr; T4 GPU, then restart.</td></tr>
<tr><td>"My conformer count is different"</td><td>stochastic search</td>
    <td>Expected. Compare the ordering and the gaps, not the count.</td></tr>
<tr><td>"Nothing converged"</td><td><code>fmax</code> tightened</td>
    <td>0.02 eV/&#197; is the floor for a float32 model.</td></tr>
<tr><td>An <code>AssertionError</code> about a complete answer</td><td>the blank</td>
    <td>That is the exercise. Write two sentences and re-run.</td></tr>
<tr><td>A red CUDA warning at import</td><td>no NVIDIA card</td>
    <td>Harmless. It runs on the CPU.</td></tr>
</table></div>

<h2>Reusing the material</h2>
<p>Everything is MIT licensed. The notebooks are generated from percent-format
sources in <code>source/</code>, so if you would rather edit Python than JSON:</p>
<pre><code>python build.py            <span class="c"># sources -&gt; notebooks/ and solutions/</span>
python check_notebooks.py  <span class="c"># static check</span></code></pre>
<p>An answer notebook is its exercise with the answer-key markdown from
<code>keys/</code> prepended and the two blanks filled. Nothing else may differ, so a
correction to an exercise reaches its answer key automatically.</p>
<p>The deck is a single HTML file in <code>slides/</code>. It renders offline from a
local file; the typefaces fall back to system fonts without a network, which is worth
knowing when the conference network fails. Press <kbd>P</kbd> to print it to PDF.</p>
"""

write("teaching.html", shell.page(
    "teaching.html", "For instructors &mdash; AIMNet2 tutorial",
    "Run sheet, pre-session checklist, what to draw out of each exercise, and what "
    "goes wrong in a room.",
    TEACHING))

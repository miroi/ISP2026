#!/usr/bin/env python
"""
Hg adsorption on Au(111) using CHGNet + D3 dispersion (tight convergence).

Reference energy for Hg is computed from a small Hg2 dimer (relaxed in a
large box, energy divided by 2). This avoids the isolated-atom graph
problem that makes a single Hg atom unreliable for CHGNet, while staying
closer to the "single atom" picture than a bulk reference would.

The Hg2 bond length is checked against the experimental value (~3.0 A
for the van der Waals dimer) as a sanity check on the reference.
"""

import os
import warnings
import numpy as np
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
from ase.io import write
from ase.calculators.mixing import SumCalculator

# Silence harmless torch.load FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*weights_only.*")
# Silence the numpy->tensor conversion warning from torch_dftd
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*Creating a tensor from a list.*")

from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator


# ============================================================
# Configuration
# ============================================================
SLAB_SIZE   = (4, 4, 4)      # 4x4x4 Au(111) slab  -> 64 Au atoms
VACUUM      = 10.0           # Angstrom
ADSORBATE   = 'Hg'           # adsorbate element
HEIGHT      = 2.0            # initial adsorbate height (Angstrom)

# Convergence
FMAX        = 0.01           # force convergence (eV/Angstrom)
MAXSTEPS    = 500            # BFGS safety limit

SITES       = ['fcc', 'hcp', 'bridge', 'ontop']

# CHGNet model configuration
CHGNET_MODEL   = "0.3.0"     # CHGNet version
CHGNET_DEVICE  = "cuda"      # "cuda" | "cpu"

# D3 dispersion configuration
D3_XC          = "pbe"       # exchange-correlation functional for D3 params
D3_DAMPING     = "bj"        # "zero" | "bj" (Becke-Johnson)
D3_DEVICE      = "cuda"      # "cuda" | "cpu"

# Hg2 reference configuration
HG2_BOX        = 12.0        # cubic box side length (Angstrom)
HG2_INIT_DIST  = 3.0         # initial Hg-Hg distance (Angstrom)
HG2_EXP_DIST   = 3.0         # experimental/CCSD(T) Hg2 bond length (Angstrom)
HG2_TOL        = 0.5         # tolerance for the sanity check (Angstrom)

OUT_STRUCT  = 'structures'
OUT_LOGS    = 'logs'
OUT_TRAJ    = 'trajectories'

# Threshold below which two adsorption energies are considered
# numerically indistinguishable (in eV).
DEGENERACY_THRESHOLD = 1e-3   # 1 meV


# ============================================================
# Helpers
# ============================================================
def ensure_dirs():
    for d in (OUT_STRUCT, OUT_LOGS, OUT_TRAJ):
        os.makedirs(d, exist_ok=True)


def max_force(atoms):
    """Return the maximum per-atom force magnitude (eV/Angstrom)."""
    return float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())


def get_chgnet_d3_calculator():
    """Initialize and return a CHGNet + D3 dispersion calculator."""
    print(f"    Initializing CHGNet + D3 "
          f"(model={CHGNET_MODEL}, device={CHGNET_DEVICE}, "
          f"xc={D3_XC}, damping={D3_DAMPING})...")

    # Base CHGNet calculator
    chgnet = CHGNet.load(model_name=CHGNET_MODEL)
    chg_calc = CHGNetCalculator(
        model=chgnet,
        use_device=CHGNET_DEVICE,
    )

    # D3 dispersion correction
    dummy_atoms = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10])
    d3_calc = TorchDFTD3Calculator(
        atoms=dummy_atoms,
        xc=D3_XC,
        damping=D3_DAMPING,
        device=D3_DEVICE,
    )

    # Combine them: energy, forces, and stress are summed
    calc = SumCalculator([chg_calc, d3_calc])

    print("    CHGNet + D3 calculator ready.")
    return calc


def relax(atoms, calc, traj_path, log_path, label=""):
    """Attach calculator, run BFGS to FMAX, return (energy, fmax, nsteps)."""
    atoms.calc = calc
    opt = BFGS(atoms,
               trajectory=traj_path,
               logfile=log_path,
               maxstep=0.2)
    opt.run(fmax=FMAX, steps=MAXSTEPS)

    E    = atoms.get_potential_energy()
    fmax = max_force(atoms)
    return E, fmax, opt.nsteps


# ============================================================
# Reference calculations
# ============================================================
def compute_E_slab(calc):
    """Relaxed energy of the clean Au(111) slab."""
    slab = fcc111('Au', size=SLAB_SIZE, vacuum=VACUUM)
    E, fmax, nsteps = relax(
        slab, calc,
        traj_path=os.path.join(OUT_TRAJ, 'slab_relax.traj'),
        log_path=os.path.join(OUT_LOGS, 'slab_relax.log'),
        label='slab',
    )
    return E, fmax, nsteps


def compute_E_atom(calc):
    """Per-atom energy of Hg from a relaxed Hg2 dimer.

    A single isolated Hg atom is unreliable for graph-based MLIPs like
    CHGNet because it has no neighbors within the graph cutoff. Using a
    small dimer instead ensures both atoms have a well-defined local
    environment, while still keeping the reference close to the
    "single atom" picture.

    Returns
    -------
    E_per_atom : float
        Total DFT-D3 energy of the relaxed Hg2 dimer, divided by 2 (eV/atom).
    info : dict
        Diagnostic info: bond length, convergence, etc.
    """
    # Build Hg2 dimer in a large cubic box
    d = HG2_INIT_DIST
    dimer = Atoms(
        [ADSORBATE, ADSORBATE],
        positions=[[0.0, 0.0, 0.0], [d, 0.0, 0.0]],
        cell=[HG2_BOX, HG2_BOX, HG2_BOX],
        pbc=True,
    )

    E_total, fmax, nsteps = relax(
        dimer, calc,
        traj_path=os.path.join(OUT_TRAJ, 'hg2_relax.traj'),
        log_path=os.path.join(OUT_LOGS, 'hg2_relax.log'),
        label='Hg2',
    )

    # Bond length after relaxation
    d_relaxed = float(np.linalg.norm(dimer.positions[1] - dimer.positions[0]))

    # Sanity check: if the dimer collapsed or dissociated, the reference
    # is not trustworthy and we should tell the user.
    bond_ok = abs(d_relaxed - HG2_EXP_DIST) < HG2_TOL

    info = {
        'E_total':    E_total,
        'E_per_atom': E_total / 2.0,
        'd_relaxed':  d_relaxed,
        'fmax':       fmax,
        'nsteps':     nsteps,
        'bond_ok':    bond_ok,
        'converged':  fmax < FMAX,
    }
    return info


# ============================================================
# Site calculation
# ============================================================
def run_site(site, calc, E_slab, E_atom):
    """Relax Hg/Au(111) at a given site; return a result dict."""
    slab = fcc111('Au', size=SLAB_SIZE, vacuum=VACUUM)
    add_adsorbate(slab, ADSORBATE, height=HEIGHT, position=site)

    E_total, fmax, nsteps = relax(
        slab, calc,
        traj_path=os.path.join(OUT_TRAJ, f'relax_{site}.traj'),
        log_path=os.path.join(OUT_LOGS,  f'relax_{site}.log'),
        label=site,
    )

    E_ads = E_total - E_slab - E_atom

    # Save relaxed geometry
    write(os.path.join(OUT_STRUCT, f'relaxed_{site}.vasp'),
          slab, format='vasp')

    return {
        'site':      site,
        'E_total':   E_total,
        'E_ads':     E_ads,
        'fmax':      fmax,
        'nsteps':    nsteps,
        'converged': fmax < FMAX,
    }


# ============================================================
# Main
# ============================================================
def main():
    ensure_dirs()

    print("=" * 72)
    print("Hg adsorption on Au(111)  --  CHGNet + D3 (tight convergence)")
    print("=" * 72)
    print(f"Slab:       Au(111) {SLAB_SIZE[0]}x{SLAB_SIZE[1]}x{SLAB_SIZE[2]}"
          f"  ({SLAB_SIZE[0]*SLAB_SIZE[1]*SLAB_SIZE[2]} Au atoms)")
    print(f"Adsorbate:  1 {ADSORBATE} atom, initial height {HEIGHT} A")
    print(f"Vacuum:     {VACUUM} A")
    print(f"Optimizer:  BFGS, fmax < {FMAX} eV/A (maxsteps={MAXSTEPS})")
    print(f"Calculator: CHGNet ({CHGNET_MODEL}) + D3 "
          f"(xc={D3_XC}, damping={D3_DAMPING})")
    print(f"Hg reference: relaxed Hg2 dimer, E/2 (box={HG2_BOX} A)")
    print(f"Degeneracy threshold: {DEGENERACY_THRESHOLD*1000:.2f} meV")
    print("-" * 72)

    # --- Initialize calculator once, share across all calculations ---
    calc = get_chgnet_d3_calculator()

    # --- Reference energies ---------------------------------
    print("\n[1/2] Reference energies")
    E_slab, fmax_slab, nsteps_slab = compute_E_slab(calc)
    slab_ok = fmax_slab < FMAX
    print(f"    E_slab           = {E_slab:.6f} eV"
          f"   (steps={nsteps_slab}, fmax={fmax_slab:.6f}"
          f"  [{'OK' if slab_ok else 'NOT CONVERGED'}])")

    hg_info = compute_E_atom(calc)
    E_atom = hg_info['E_per_atom']
    print(f"    E(Hg2)/2         = {E_atom:.6f} eV/atom"
          f"   (steps={hg_info['nsteps']}, "
          f"fmax={hg_info['fmax']:.6f}, "
          f"d(Hg-Hg)={hg_info['d_relaxed']:.3f} A"
          f"  [{'bond OK' if hg_info['bond_ok'] else 'BOND SUSPECT'}])")

    if not slab_ok:
        print("\n    WARNING: slab did not reach FMAX. "
              "E_slab (and thus every E_ads) is unreliable.")
    if not hg_info['converged']:
        print("\n    WARNING: Hg2 reference did not converge. "
              "E_atom (and thus every E_ads) is unreliable.")
    if not hg_info['bond_ok']:
        print(f"\n    WARNING: relaxed Hg-Hg bond length "
              f"({hg_info['d_relaxed']:.3f} A) differs from the expected "
              f"~{HG2_EXP_DIST} A by more than {HG2_TOL} A. "
              f"The Hg reference may not be physical.")

    # --- Site sweep -----------------------------------------
    print("\n[2/2] Adsorption sites")
    results = []
    for site in SITES:
        print(f"\n  -> {site}")
        r = run_site(site, calc, E_slab, E_atom)
        results.append(r)
        status = "OK" if r['converged'] else "NOT CONVERGED"
        print(f"     E_total = {r['E_total']:.6f} eV")
        print(f"     E_ads   = {r['E_ads']:.6f} eV")
        print(f"     fmax    = {r['fmax']:.6f} eV/A   "
              f"[{status}, {r['nsteps']} steps]")

    # --- Summary table --------------------------------------
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    header = (f"{'Site':<8s}{'E_total (eV)':>16s}"
              f"{'E_ads (eV)':>16s}{'fmax (eV/A)':>16s}{'Steps':>8s}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['site']:<8s}"
              f"{r['E_total']:>16.6f}"
              f"{r['E_ads']:>16.6f}"
              f"{r['fmax']:>16.6f}"
              f"{r['nsteps']:>8d}")

    # --- Ranking --------------------------------------------
    ranked = sorted(results, key=lambda r: r['E_ads'])
    print("\nSite preference (most stable first):")
    for i, r in enumerate(ranked, 1):
        marker = "  <-- most stable" if i == 1 else ""
        print(f"  {i}. {r['site']:<8s}  E_ads = {r['E_ads']:.6f} eV{marker}")

    # --- Numerical-resolution diagnostic --------------------
    print("\nNumerical resolution check "
          f"(threshold = {DEGENERACY_THRESHOLD*1000:.2f} meV):")
    if len(ranked) >= 2:
        gap = ranked[1]['E_ads'] - ranked[0]['E_ads']
        if gap < DEGENERACY_THRESHOLD:
            print(f"  {ranked[0]['site']} vs {ranked[1]['site']}: "
                  f"gap = {gap*1000:.2f} meV  "
                  f"-> NOT RESOLVABLE, treat as degenerate")
        else:
            print(f"  {ranked[0]['site']} vs {ranked[1]['site']}: "
                  f"gap = {gap*1000:.2f} meV  -> resolved")
    for i in range(1, len(ranked) - 1):
        gap = ranked[i+1]['E_ads'] - ranked[i]['E_ads']
        tag = "resolved" if gap >= DEGENERACY_THRESHOLD else "NOT RESOLVABLE"
        print(f"  {ranked[i]['site']} vs {ranked[i+1]['site']}: "
              f"gap = {gap*1000:.2f} meV  -> {tag}")

    # --- Convergence summary --------------------------------
    all_ok = (all(r['converged'] for r in results)
              and slab_ok
              and hg_info['converged'])
    if all_ok:
        print("\nAll calculations converged.\n")
    else:
        print("\nWARNING: not all calculations converged!\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Hg adsorption on Au(111) using CHGNet + D3 dispersion (tight convergence).

This script combines the CHGNet universal MLIP with a D3 dispersion
correction via the torch-dftd library and ASE's SumCalculator.

Note on reference energies: The isolated Hg atom reference is unreliable
for graph-based MLIPs like CHGNet due to the lack of neighbors within the
graph cutoff. The adsorption energies reported here should be interpreted
as relative rankings. For absolute values, a bulk/cohesive reference is
recommended.
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
    # Note: TorchDFTD3Calculator needs a template atoms object to initialize
    # some internal tensors. We create a dummy and it will be updated
    # when the calculator is actually used on real atoms.
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
    """Energy of an isolated Hg atom in a large box (no relaxation).

    WARNING: This reference is problematic for graph-based MLIPs like CHGNet
    because an isolated atom has no neighbors within the graph cutoff.
    The value is an extrapolation artifact. Use with caution.
    """
    atom = Atoms(ADSORBATE,
                 positions=[[0, 0, 0]],
                 cell=[10, 10, 10],
                 pbc=False)
    atom.calc = calc
    return atom.get_potential_energy()


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
    print(f"Degeneracy threshold: {DEGENERACY_THRESHOLD*1000:.2f} meV")
    print("-" * 72)

    # --- Initialize calculator once, share across all calculations ---
    calc = get_chgnet_d3_calculator()

    # --- Reference energies ---------------------------------
    print("\n[1/2] Reference energies")
    E_slab, fmax_slab, nsteps_slab = compute_E_slab(calc)
    E_atom = compute_E_atom(calc)
    slab_ok = fmax_slab < FMAX
    print(f"    E_slab           = {E_slab:.6f} eV"
          f"   (steps={nsteps_slab}, fmax={fmax_slab:.6f}"
          f"  [{'OK' if slab_ok else 'NOT CONVERGED'}])")
    print(f"    E_atom({ADSORBATE:<2s})     = {E_atom:.6f} eV"
          f"   [WARNING: unreliable for graph MLIPs]")

    if not slab_ok:
        print("\n    WARNING: slab did not reach FMAX. "
              "E_slab (and thus every E_ads) is unreliable.")

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
    if all(r['converged'] for r in results) and slab_ok:
        print("\nAll calculations converged.\n")
    else:
        print("\nWARNING: not all calculations converged!\n")


if __name__ == "__main__":
    main()

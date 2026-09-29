#!/usr/bin/env python
"""
Hg13 cluster adsorption on Au(111) using MACE-MP-0 + D3.

Computes:
    E_ads = E_total - (E_slab + E_cluster)

where:
    E_cluster = relaxed isolated Hg13 icosahedron in vacuum
    E_slab    = relaxed clean Au(111) slab (bottom layers fixed)
    E_total   = relaxed Hg13/Au(111) system
"""

import os
import warnings
import numpy as np
from ase.build import fcc111, add_adsorbate
from ase.cluster import Icosahedron
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.io import read

# Silence harmless torch warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*weights_only.*")

from mace.calculators import mace_mp


# ============================================================
# Configuration
# ============================================================
FMAX          = 0.05          # force convergence (eV/A)
MAXSTEPS      = 500
SLAB_SIZE     = (4, 4, 4)     # 4 layers -> 2 fixed, 2 mobile
VACUUM        = 10.0
CLUSTER_SHELLS = 2            # noshells=2 -> Hg13
HG_LATTICE    = 3.0           # initial icosahedron scaling
ADS_HEIGHT    = 2.5           # initial height above surface (A)
ADS_SITE      = 'fcc'

MACE_MODEL    = "medium"      # "small" | "medium" | "large"
MACE_DEVICE   = "cuda"
MACE_DTYPE    = "float64"     # REQUIRED for geometry optimization
DISPERSION    = True          # D3 dispersion (needs torch-dftd)


# ============================================================
# Calculator (shared)
# ============================================================
print(f"Initializing MACE ({MACE_MODEL}, {MACE_DTYPE}, "
      f"{MACE_DEVICE}, dispersion={DISPERSION})...")
calc = mace_mp(
    model=MACE_MODEL,
    dispersion=DISPERSION,
    default_dtype=MACE_DTYPE,
    device=MACE_DEVICE,
)
print("MACE calculator ready.\n")


# ============================================================
# Helpers
# ============================================================
def max_force(atoms):
    return float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())


def get_relaxed_energy(atoms, name):
    """Relax and return (energy, fmax, nsteps)."""
    atoms.calc = calc
    print(f"Optimizing {name}...")
    dyn = BFGS(atoms,
               trajectory=f'{name}.traj',
               logfile=f'{name}.log',
               maxstep=0.2)
    dyn.run(fmax=FMAX, steps=MAXSTEPS)

    E    = atoms.get_potential_energy()
    fmax = max_force(atoms)

    status = "OK" if fmax < FMAX else "NOT CONVERGED"
    print(f"  {name}: E = {E:.6f} eV, "
          f"fmax = {fmax:.6f} eV/A  [{status}, {dyn.nsteps} steps]")
    return E, fmax, dyn.nsteps


def fix_bottom_layers(atoms, n_fixed=2):
    """Fix the bottom n_fixed layers (by tag, lowest tags = bottom)."""
    mask = [atom.tag <= n_fixed for atom in atoms]
    atoms.set_constraint(FixAtoms(mask=mask))
    return sum(mask)


# ============================================================
# 1. Isolated Hg13 cluster
# ============================================================
print("=" * 68)
print("[1/3] Isolated Hg13 cluster")
print("=" * 68)
hg13_iso = Icosahedron('Hg', noshells=CLUSTER_SHELLS,
                       latticeconstant=HG_LATTICE)
hg13_iso.center(vacuum=VACUUM)
n_hg = len(hg13_iso)
print(f"  Cluster size: {n_hg} Hg atoms")
E_cluster, fmax_c, nsteps_c = get_relaxed_energy(hg13_iso, "cluster_iso")


# ============================================================
# 2. Clean Au(111) slab
# ============================================================
print("\n" + "=" * 68)
print("[2/3] Clean Au(111) slab")
print("=" * 68)
slab_clean = fcc111('Au', size=SLAB_SIZE, vacuum=VACUUM)
n_fixed = fix_bottom_layers(slab_clean, n_fixed=2)
print(f"  Slab: {len(slab_clean)} Au atoms "
      f"({n_fixed} fixed, {len(slab_clean) - n_fixed} mobile)")
E_slab, fmax_s, nsteps_s = get_relaxed_energy(slab_clean, "slab_clean")


# ============================================================
# 3. Combined system: Hg13 on Au(111)
# ============================================================
print("\n" + "=" * 68)
print("[3/3] Hg13 / Au(111)")
print("=" * 68)
system = fcc111('Au', size=SLAB_SIZE, vacuum=VACUUM)
hg13_ads = Icosahedron('Hg', noshells=CLUSTER_SHELLS,
                       latticeconstant=HG_LATTICE)
add_adsorbate(system, hg13_ads, height=ADS_HEIGHT,
              position=ADS_SITE)
fix_bottom_layers(system, n_fixed=2)
print(f"  System: {len(system)} atoms "
      f"({len(slab_clean)} Au + {len(hg13_ads)} Hg), "
      f"site={ADS_SITE}")
E_total, fmax_t, nsteps_t = get_relaxed_energy(system, "total_system")


# ============================================================
# 4. Adsorption energy
# ============================================================
E_ads = E_total - (E_slab + E_cluster)
E_ads_per_atom = E_ads / n_hg

print("\n" + "=" * 68)
print("Result")
print("=" * 68)
print(f"  E_total          = {E_total:12.6f} eV")
print(f"  E_slab           = {E_slab:12.6f} eV")
print(f"  E_cluster        = {E_cluster:12.6f} eV")
print(f"  -----------------------------------------")
print(f"  E_ads (cluster)  = {E_ads:12.6f} eV")
print(f"  E_ads / Hg atom  = {E_ads_per_atom:12.6f} eV")
print("=" * 68)

if E_ads < 0:
    print("The adsorption is EXOTHERMIC (stable).")
else:
    print("The adsorption is ENDOTHERMIC (unstable).")

# Convergence summary
all_ok = all(f < FMAX for f in (fmax_c, fmax_s, fmax_t))
print(f"\nConvergence: "
      f"{'all OK' if all_ok else 'WARNING — some runs did not converge'}")

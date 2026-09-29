#!/usr/bin/env python
"""
Hg13 cluster adsorption on Au(111) using MACE-MP-0 + D3.

Fully unconstrained relaxation: all Au and Hg atoms are mobile.
Computes:
    E_ads = E_total - (E_slab + E_cluster)

Includes geometry diagnostics:
    - minimum Hg-Au distance at start and end
    - cluster integrity check (radius of gyration)
"""

import os
import warnings
import numpy as np
from ase.build import fcc111
from ase.cluster import Icosahedron
from ase.optimize import BFGS
from ase.io import read

warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*weights_only.*")

from mace.calculators import mace_mp


# ============================================================
# Configuration
# ============================================================
FMAX           = 0.05          # force convergence (eV/A)
MAXSTEPS       = 1000          # larger budget, no constraint on Au
SLAB_SIZE      = (4, 4, 4)     # 64 Au atoms, all mobile
VACUUM         = 10.0
CLUSTER_SHELLS = 2             # noshells=2 -> Hg13
HG_LATTICE     = 3.0
ADS_HEIGHT     = 2.5           # initial gap between cluster and top Au layer
MACE_MODEL     = "medium"
MACE_DEVICE    = "cuda"
MACE_DTYPE     = "float64"
DISPERSION     = True

MIN_ALLOWED_HG_AU = 2.0        # Å — below this, atoms overlap


# ============================================================
# Calculator
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


def min_hg_au_distance(atoms):
    """Return the minimum distance between any Hg and any Au atom."""
    hg = [a for a in atoms if a.symbol == 'Hg']
    au = [a for a in atoms if a.symbol == 'Au']
    return min(
        float(np.linalg.norm(h.position - a.position))
        for h in hg for a in au
    )


def radius_of_gyration(atoms, symbol='Hg'):
    """Rg of a given element — a compactness metric for the cluster."""
    pos = np.array([a.position for a in atoms if a.symbol == symbol])
    com = pos.mean(axis=0)
    return float(np.sqrt(((pos - com) ** 2).sum(axis=1).mean()))


def get_relaxed_energy(atoms, name):
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
rg_iso = radius_of_gyration(hg13_iso, 'Hg')
print(f"  Cluster Rg (isolated): {rg_iso:.3f} Å")


# ============================================================
# 2. Clean Au(111) slab
# ============================================================
print("\n" + "=" * 68)
print("[2/3] Clean Au(111) slab (fully unconstrained)")
print("=" * 68)
slab_clean = fcc111('Au', size=SLAB_SIZE, vacuum=VACUUM)
print(f"  Slab: {len(slab_clean)} Au atoms, no constraints")
E_slab, fmax_s, nsteps_s = get_relaxed_energy(slab_clean, "slab_clean")


# ============================================================
# 3. Combined system: Hg13 on Au(111), manual placement
# ============================================================
print("\n" + "=" * 68)
print("[3/3] Hg13 / Au(111)  (manual placement, no constraints)")
print("=" * 68)

# Build fresh, unconstrained slab
slab = fcc111('Au', size=SLAB_SIZE, vacuum=VACUUM)

# Build cluster, centered at origin, no vacuum padding
hg13 = Icosahedron('Hg', noshells=CLUSTER_SHELLS,
                   latticeconstant=HG_LATTICE)
hg13.center(vacuum=0)

# Position cluster above the slab's top layer
z_top         = slab.positions[:, 2].max()
z_cluster_min = hg13.positions[:, 2].min()
shift_z       = z_top + ADS_HEIGHT - z_cluster_min

# Place over cell center (approximate fcc hollow)
shift_x = slab.cell[0, 0] / 2
shift_y = slab.cell[1, 1] / 2

hg13.translate([shift_x, shift_y, shift_z])

# Combine
system = slab + hg13

# --- Diagnostics BEFORE relaxation ---
dmin_start = min_hg_au_distance(system)
print(f"  Atoms: {len(system)} ({len(slab)} Au + {len(hg13)} Hg)")
print(f"  Min Hg-Au distance (start): {dmin_start:.3f} Å")
assert dmin_start > MIN_ALLOWED_HG_AU, (
    f"Atoms overlap at start: dmin = {dmin_start:.3f} Å"
)

# --- Relax ---
E_total, fmax_t, nsteps_t = get_relaxed_energy(system, "total_system")


# ============================================================
# 4. Post-relaxation geometry diagnostics
# ============================================================
print("\n" + "=" * 68)
print("Geometry diagnostics (final frame)")
print("=" * 68)

final = read('total_system.traj', index=-1)

dmin_end = min_hg_au_distance(final)
rg_ads   = radius_of_gyration(final, 'Hg')

print(f"  Min Hg-Au distance (end):  {dmin_end:.3f} Å")
print(f"  Cluster Rg (adsorbed):     {rg_ads:.3f} Å")
print(f"  Cluster Rg (isolated):     {rg_iso:.3f} Å")
print(f"  Rg ratio (ads/iso):        {rg_ads / rg_iso:.3f}")

# Interpretation
if dmin_end < MIN_ALLOWED_HG_AU:
    print("  WARNING: Hg and Au atoms are still overlapping.")
elif dmin_end > 4.0:
    print("  NOTE: Hg cluster may have desorbed (dmin > 4 Å).")
else:
    print("  Hg cluster is in contact with the surface.")

if rg_ads / rg_iso > 1.3:
    print("  WARNING: cluster has spread out "
          "(Rg increased by >30%) — may have deformed.")
elif rg_ads / rg_iso < 0.7:
    print("  WARNING: cluster has contracted significantly.")
else:
    print("  Cluster shape is preserved.")


# ============================================================
# 5. Adsorption energy
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

# Sanity guards
if E_total > 0:
    print("ERROR: E_total is positive — combined system is unphysical.")
    print("       Check the starting geometry (total_system.traj@0).")
elif E_ads > 0:
    print("The adsorption is ENDOTHERMIC (unstable).")
    print("Verify: did the cluster stay on the surface? "
          "See diagnostics above.")
else:
    print("The adsorption is EXOTHERMIC (stable).")

all_ok = all(f < FMAX for f in (fmax_c, fmax_s, fmax_t))
print(f"\nConvergence: "
      f"{'all OK' if all_ok else 'WARNING — some runs did not converge'}")

#!/usr/bin/env python
"""
Small Hg cluster (Hg4 or Hg7) adsorption on Au(111)
using MACE-MP-0 + D3.

Usage:
    python hg_cluster_on_gold.py --cluster hg7
    python hg_cluster_on_gold.py --cluster hg4

Computes:
    E_ads = E_total - (E_slab + E_cluster)

Outputs (VASP POSCAR, openable in VESTA):
    cluster_initial.vasp, cluster_final.vasp
    slab_initial.vasp,    slab_final.vasp
    total_initial.vasp,   total_final.vasp

Diagnostics:
    - cell-size and vacuum checks
    - Hg-Au and Hg-Hg start-distance assertions
    - lateral COM drift during relaxation
    - minimum Hg-Au contact distance (end)
    - minimum Hg-Hg image distance (periodic coupling)
    - cluster Rg and diameter (isolated vs. adsorbed)
    - lateral footprint (wetting descriptor)
"""

import argparse
import warnings
import numpy as np
from ase import Atom, Atoms
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
from ase.io import read, write

warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*numpy.ndarrays.*")

from mace.calculators import mace_mp


# ============================================================
# Command-line arguments
# ============================================================
parser = argparse.ArgumentParser(
    description="Small Hg cluster adsorption on Au(111) with MACE+D3"
)
parser.add_argument(
    "--cluster", choices=["hg4", "hg7"], default="hg7",
    help="Which Hg cluster to adsorb (default: hg7)"
)
args = parser.parse_args()

CLUSTER_KIND = args.cluster


# ============================================================
# Configuration
# ============================================================
FMAX           = 0.05
MAXSTEPS       = 1500
SLAB_SIZE      = (5, 5, 4)
VACUUM         = 20.0
ADS_HEIGHT     = 2.5
MACE_MODEL     = "medium"
MACE_DEVICE    = "cuda"
MACE_DTYPE     = "float64"
DISPERSION     = True

MIN_ALLOWED_HG_AU  = 2.0
MIN_ALLOWED_HG_HG  = 2.5
DRIFT_WARN_THRESH  = 1.0
IMAGE_WARN_HG_HG   = 6.0
IMAGE_WARN_RATIO   = 3.0


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
# Cluster builders
# ============================================================
def build_hg4():
    """Hg4 tetrahedron with Hg-Hg ~3.0 Å."""
    a = 3.0
    positions = np.array([
        [0, 0, 0],
        [a, 0, 0],
        [a/2, a*np.sqrt(3)/2, 0],
        [a/2, a*np.sqrt(3)/6, a*np.sqrt(2/3)],
    ])
    return Atoms('Hg4', positions=positions,
                 cell=[20, 20, 20], pbc=False)


def build_hg7():
    """Pentagonal bipyramid Hg7."""
    R_eq = 1.65
    R_ax = 1.60
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    eq = np.array([[R_eq*np.cos(a), R_eq*np.sin(a), 0.0]
                   for a in angles])
    ax = np.array([[0, 0,  R_ax],
                   [0, 0, -R_ax]])
    positions = np.vstack([eq, ax])
    return Atoms('Hg7', positions=positions,
                 cell=[20, 20, 20], pbc=False)


CLUSTER_BUILDERS = {
    'hg4': build_hg4,
    'hg7': build_hg7,
}


def build_cluster(kind):
    return CLUSTER_BUILDERS[kind]()


# ============================================================
# Geometry helpers
# ============================================================
def max_force(atoms):
    return float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())


def min_hg_au_distance(atoms):
    hg = [a for a in atoms if a.symbol == 'Hg']
    au = [a for a in atoms if a.symbol == 'Au']
    return min(
        float(np.linalg.norm(h.position - a.position))
        for h in hg for a in au
    )


def min_hg_hg_distance(atoms):
    hg = [a for a in atoms if a.symbol == 'Hg']
    if len(hg) < 2:
        return np.inf
    return min(
        float(np.linalg.norm(hg[i].position - hg[j].position))
        for i in range(len(hg))
        for j in range(i+1, len(hg))
    )


def min_hg_hg_image_distance(atoms):
    hg_idx = [a.index for a in atoms if a.symbol == 'Hg']
    if len(hg_idx) < 2:
        return None
    D = atoms.get_all_distances(mic=True)
    sub = D[np.ix_(hg_idx, hg_idx)].copy()
    np.fill_diagonal(sub, np.inf)
    return float(sub.min())


def radius_of_gyration(atoms, symbol='Hg'):
    pos = np.array([a.position for a in atoms if a.symbol == symbol])
    com = pos.mean(axis=0)
    return float(np.sqrt(((pos - com) ** 2).sum(axis=1).mean()))


def cluster_diameter(atoms, symbol='Hg'):
    pos = np.array([a.position for a in atoms if a.symbol == symbol])
    if len(pos) < 2:
        return 0.0
    dmax = 0.0
    for i in range(len(pos)):
        d = np.linalg.norm(pos[i+1:] - pos[i], axis=1)
        if len(d):
            dmax = max(dmax, d.max())
    return float(dmax)


def lateral_footprint(atoms, symbol='Hg'):
    pos = np.array([a.position for a in atoms if a.symbol == symbol])
    if len(pos) < 2:
        return 0.0
    dx = pos[:, 0].max() - pos[:, 0].min()
    dy = pos[:, 1].max() - pos[:, 1].min()
    return float(np.sqrt(dx*dx + dy*dy))


def com_offset(atoms):
    hg_com = np.mean([a.position for a in atoms if a.symbol == 'Hg'],
                     axis=0)
    au_com = np.mean([a.position for a in atoms if a.symbol == 'Au'],
                     axis=0)
    return hg_com[:2] - au_com[:2]


def fcc_hollow_position(slab):
    probe = slab.copy()
    add_adsorbate(probe, Atom('He'), height=1.0, position='fcc')
    return probe.positions[-1][:2].copy()


# ============================================================
# VESTA-friendly output
# ============================================================
def save_vasp(atoms, filename):
    """
    Write a VASP POSCAR file readable by VESTA.

    VESTA opens files with the .vasp extension directly.
    We use ASE's 'vasp' format, which produces a standard POSCAR.
    """
    write(filename, atoms, format='vasp', vasp5=True, direct=False)
    print(f"    wrote {filename}")


# ============================================================
# Checks
# ============================================================
def report_cell_checks(slab, cluster):
    cell = slab.cell
    a_avg = 0.5 * (np.linalg.norm(cell[0]) + np.linalg.norm(cell[1]))
    d_cluster = cluster_diameter(cluster, 'Hg')
    ratio = a_avg / d_cluster
    edge_gap = a_avg - d_cluster

    print(f"  --- Cell size checks ---")
    print(f"  Cell side (avg in-plane):   {a_avg:.2f} Å")
    print(f"  Cluster diameter:           {d_cluster:.2f} Å")
    print(f"  Cell/cluster ratio:         {ratio:.2f}")
    print(f"  Edge-to-edge gap:           {edge_gap:6.2f} Å")
    if ratio < IMAGE_WARN_RATIO:
        print(f"  WARNING: cell/cluster ratio < {IMAGE_WARN_RATIO:.1f}")
    else:
        print(f"  Cell is large enough.")

    z_slab = slab.positions[:, 2].max() - slab.positions[:, 2].min()
    z_cluster = cluster.positions[:, 2].max() - cluster.positions[:, 2].min()
    cell_z = np.linalg.norm(cell[2])
    eff_vac = cell_z - z_slab - z_cluster

    print(f"  --- Vacuum checks ---")
    print(f"  Cell height (z):            {cell_z:.2f} Å")
    print(f"  Slab thickness:             {z_slab:.2f} Å")
    print(f"  Cluster thickness:          {z_cluster:.2f} Å")
    print(f"  Effective vacuum:           {eff_vac:.2f} Å")
    if eff_vac < 10.0:
        print(f"  WARNING: effective vacuum < 10 Å.")
    else:
        print(f"  Vacuum is sufficient.")


def report_image_checks(final):
    cell = final.cell
    a_avg = 0.5 * (np.linalg.norm(cell[0]) + np.linalg.norm(cell[1]))
    d_ads = cluster_diameter(final, 'Hg')
    d_hg_hg = min_hg_hg_image_distance(final)
    edge_gap = a_avg - d_ads

    print(f"  --- Post-relaxation image checks ---")
    print(f"  Cell side (avg in-plane):   {a_avg:.2f} Å")
    print(f"  Adsorbed cluster diameter:  {d_ads:.2f} Å")
    print(f"  Edge-to-edge gap:           {edge_gap:6.2f} Å")
    print(f"  Min Hg-Hg image distance:   {d_hg_hg:.3f} Å")

    if d_hg_hg < IMAGE_WARN_HG_HG:
        print(f"  WARNING: Hg-Hg image distance < "
              f"{IMAGE_WARN_HG_HG:.1f} Å — periodic images may interact.")
    else:
        print(f"  Hg-Hg image distance is large enough.")


# ============================================================
# Relaxation driver
# ============================================================
def get_relaxed_energy(atoms, name, monitor_com=False):
    atoms.calc = calc
    print(f"Optimizing {name}...")

    com_start = com_offset(atoms) if monitor_com else None

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

    if monitor_com:
        com_end = com_offset(atoms)
        drift = np.linalg.norm(com_end - com_start)
        print(f"  Lateral COM drift: {drift:.3f} Å "
              f"(dx = {com_end[0]-com_start[0]:+.3f}, "
              f"dy = {com_end[1]-com_start[1]:+.3f})")
        if drift > DRIFT_WARN_THRESH:
            print(f"  NOTE: cluster migrated > {DRIFT_WARN_THRESH} Å.")

    return E, fmax, dyn.nsteps


# ============================================================
# 1. Isolated cluster
# ============================================================
print("=" * 68)
print(f"[1/3] Isolated {CLUSTER_KIND.upper()} cluster")
print("=" * 68)

cluster_iso = build_cluster(CLUSTER_KIND)
cluster_iso.center(vacuum=VACUUM)
n_hg = len(cluster_iso)
print(f"  Cluster size: {n_hg} Hg atoms")

# --- VASP output: initial ---
save_vasp(cluster_iso, "cluster_initial.vasp")

E_cluster, fmax_c, nsteps_c = get_relaxed_energy(cluster_iso, "cluster_iso")

# --- VASP output: final ---
save_vasp(cluster_iso, "cluster_final.vasp")

rg_iso = radius_of_gyration(cluster_iso, 'Hg')
d_iso  = cluster_diameter(cluster_iso, 'Hg')
fp_iso = lateral_footprint(cluster_iso, 'Hg')
print(f"  Cluster Rg (isolated):       {rg_iso:.3f} Å")
print(f"  Cluster diameter (iso):      {d_iso:.3f} Å")
print(f"  Lateral footprint (iso):     {fp_iso:.3f} Å")


# ============================================================
# 2. Clean Au(111) slab
# ============================================================
print("\n" + "=" * 68)
print(f"[2/3] Clean Au(111) slab {SLAB_SIZE}")
print("=" * 68)

slab_clean = fcc111('Au', size=SLAB_SIZE, vacuum=VACUUM)
print(f"  Slab: {len(slab_clean)} Au atoms, no constraints")

# --- VASP output: initial ---
save_vasp(slab_clean, "slab_initial.vasp")

E_slab, fmax_s, nsteps_s = get_relaxed_energy(slab_clean, "slab_clean")

# --- VASP output: final ---
save_vasp(slab_clean, "slab_final.vasp")


# ============================================================
# 3. Combined system
# ============================================================
print("\n" + "=" * 68)
print(f"[3/3] {CLUSTER_KIND.upper()} / Au(111) {SLAB_SIZE}")
print("=" * 68)

slab = fcc111('Au', size=SLAB_SIZE, vacuum=VACUUM)

cluster = build_cluster(CLUSTER_KIND)
cluster.center(vacuum=0)

report_cell_checks(slab, cluster)
print()

fcc_xy = fcc_hollow_position(slab)

# Vertical placement
z_top         = slab.positions[:, 2].max()
z_cluster_min = cluster.positions[:, 2].min()
shift_z       = z_top + ADS_HEIGHT - z_cluster_min

# Lateral placement over fcc hollow
hg_com_xy = np.mean(cluster.positions[:, :2], axis=0)
shift_xy  = fcc_xy - hg_com_xy

cluster.translate([shift_xy[0], shift_xy[1], shift_z])

system = slab + cluster

# --- Diagnostics BEFORE relaxation ---
dmin_hg_au = min_hg_au_distance(system)
dmin_hg_hg = min_hg_hg_distance(system)
print(f"  Atoms: {len(system)} ({len(slab)} Au + {len(cluster)} Hg)")
print(f"  Min Hg-Au distance (start):  {dmin_hg_au:.3f} Å")
print(f"  Min Hg-Hg distance (start):  {dmin_hg_hg:.3f} Å")

assert dmin_hg_au > MIN_ALLOWED_HG_AU, (
    f"Hg-Au overlap at start: {dmin_hg_au:.3f} Å"
)
assert dmin_hg_hg > MIN_ALLOWED_HG_HG, (
    f"Hg-Hg overlap at start: {dmin_hg_hg:.3f} Å"
)

hg_com_placed = np.mean([a.position for a in system if a.symbol == 'Hg'],
                        axis=0)
d_fcc = np.linalg.norm(hg_com_placed[:2] - fcc_xy)
print(f"  Hg COM distance from fcc site: {d_fcc:.3f} Å")
assert d_fcc < 0.5, (
    f"Cluster not placed over fcc: offset = {d_fcc:.3f} Å"
)

# --- VASP output: initial ---
save_vasp(system, "total_initial.vasp")

# Pre-relaxation energy guard
system.calc = calc
E_initial = system.get_potential_energy()
print(f"  Initial energy (step 0):     {E_initial:.3f} eV")
if E_initial > 0:
    raise RuntimeError(
        f"Initial energy is positive ({E_initial:.2f} eV) — "
        "geometry has severe overlap."
    )

# Relax
E_total, fmax_t, nsteps_t = get_relaxed_energy(
    system, "total_system", monitor_com=True
)

# --- VASP output: final ---
save_vasp(system, "total_final.vasp")


# ============================================================
# 4. Post-relaxation diagnostics
# ============================================================
print("\n" + "=" * 68)
print("Geometry diagnostics (final frame)")
print("=" * 68)

final = read('total_system.traj', index=-1)

dmin_hg_au_end = min_hg_au_distance(final)
dmin_hg_hg_end = min_hg_hg_distance(final)
rg_ads   = radius_of_gyration(final, 'Hg')
d_ads    = cluster_diameter(final, 'Hg')
fp_ads   = lateral_footprint(final, 'Hg')

print(f"  Min Hg-Au distance (end):    {dmin_hg_au_end:.3f} Å")
print(f"  Min Hg-Hg distance (end):    {dmin_hg_hg_end:.3f} Å")
print(f"  Cluster Rg (adsorbed):       {rg_ads:.3f} Å")
print(f"  Cluster Rg (isolated):       {rg_iso:.3f} Å")
print(f"  Rg ratio (ads/iso):          {rg_ads / rg_iso:.3f}")
print(f"  Cluster diameter (ads):      {d_ads:.3f} Å")
print(f"  Cluster diameter (iso):      {d_iso:.3f} Å")
print(f"  Diameter ratio (ads/iso):    {d_ads / d_iso:.3f}")
print(f"  Lateral footprint (ads):     {fp_ads:.3f} Å")
print(f"  Lateral footprint (iso):     {fp_iso:.3f} Å")
print(f"  Footprint ratio (ads/iso):   {fp_ads / fp_iso:.3f}")

print()
report_image_checks(final)

# Contact / shape checks
print()
if dmin_hg_au_end < MIN_ALLOWED_HG_AU:
    print("  WARNING: Hg and Au atoms are still overlapping.")
elif dmin_hg_au_end > 4.0:
    print("  NOTE: cluster may have desorbed (dmin > 4 Å).")
else:
    print("  Hg cluster is in contact with the surface.")

if dmin_hg_hg_end < MIN_ALLOWED_HG_HG:
    print("  WARNING: intra-cluster Hg-Hg distance below threshold — "
          "cluster may have partially dissociated.")

if rg_ads / rg_iso > 1.3:
    print("  WARNING: cluster has spread out (>30% Rg increase).")
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
print(f"Result  ({CLUSTER_KIND.upper()} on Au(111))")
print("=" * 68)
print(f"  Slab:             {SLAB_SIZE} ({len(slab)} Au atoms)")
print(f"  Vacuum:           {VACUUM} Å")
print(f"  Cluster:          {CLUSTER_KIND.upper()} ({n_hg} Hg atoms)")
print(f"  E_total          = {E_total:12.6f} eV")
print(f"  E_slab           = {E_slab:12.6f} eV")
print(f"  E_cluster        = {E_cluster:12.6f} eV")
print(f"  -----------------------------------------")
print(f"  E_ads (cluster)  = {E_ads:12.6f} eV")
print(f"  E_ads / Hg atom  = {E_ads_per_atom:12.6f} eV")
print("=" * 68)

if E_ads > 0:
    print("The adsorption is ENDOTHERMIC (unstable).")
else:
    print("The adsorption is EXOTHERMIC (stable).")

all_ok = all(f < FMAX for f in (fmax_c, fmax_s, fmax_t))
print(f"\nConvergence: "
      f"{'all OK' if all_ok else 'WARNING — some runs did not converge'}")

print("\nVESTA-ready output files:")
for f in ["cluster_initial.vasp", "cluster_final.vasp",
          "slab_initial.vasp",    "slab_final.vasp",
          "total_initial.vasp",   "total_final.vasp"]:
    print(f"  {f}")

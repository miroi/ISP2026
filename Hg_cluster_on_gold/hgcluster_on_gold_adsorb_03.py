#!/usr/bin/env python
"""
Hg13 cluster adsorption on Au(111) using MACE-MP-0 + D3.

Fully unconstrained relaxation: all Au and Hg atoms are mobile.

Computes:
    E_ads = E_total - (E_slab + E_cluster)

Diagnostics included:
    - correct fcc-site placement using ASE's site finder
    - registry check at start and end (fcc/hcp/bridge/top)
    - lateral drift monitoring during relaxation
    - minimum Hg-Au distance at start and end
    - minimum Hg-Hg distance across periodic images (MIC)
    - cell-size and vacuum checks (using adsorbed diameter too)
    - cluster integrity check (radius of gyration, diameter)

Output:
    - POSCAR files for every system (isolated cluster, slab, combined)
    - POSCAR files for every frame of the combined-system trajectory
    - trajectory (.traj) and log (.log) files for each relaxation
"""

import os
import warnings
import numpy as np
from ase import Atom
from ase.build import fcc111, add_adsorbate
from ase.cluster import Icosahedron
from ase.optimize import BFGS
from ase.io import read, write

warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*numpy.ndarrays.*")

from mace.calculators import mace_mp


# ============================================================
# Configuration
# ============================================================
FMAX           = 0.05
MAXSTEPS       = 1500
SLAB_SIZE      = (5, 5, 4)      # 100 Au atoms, ~14.6 Å per side
VACUUM         = 20.0           # Å
CLUSTER_SHELLS = 2              # noshells=2 -> Hg13
HG_LATTICE     = 3.0
ADS_HEIGHT     = 2.5
MACE_MODEL     = "medium"
MACE_DEVICE    = "cuda"
MACE_DTYPE     = "float64"
DISPERSION     = True

MIN_ALLOWED_HG_AU  = 2.0        # Å — below this, atoms overlap
DRIFT_WARN_THRESH  = 1.0        # Å — warn if Hg COM moves more than this
IMAGE_WARN_HG_HG   = 6.0        # Å — warn if Hg-Hg image distance below
IMAGE_WARN_RATIO   = 3.0        # cell side / cluster diameter


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
# Geometry helpers
# ============================================================
def max_force(atoms):
    return float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())


def min_hg_au_distance(atoms):
    """Minimum distance between any Hg and any Au atom."""
    hg = [a for a in atoms if a.symbol == 'Hg']
    au = [a for a in atoms if a.symbol == 'Au']
    return min(
        float(np.linalg.norm(h.position - a.position))
        for h in hg for a in au
    )


def min_hg_hg_image_distance(atoms):
    """
    Minimum Hg-Hg distance including periodic images
    (minimum image convention).
    """
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
    """Maximum pairwise distance within the cluster (no MIC)."""
    pos = np.array([a.position for a in atoms if a.symbol == symbol])
    if len(pos) < 2:
        return 0.0
    dmax = 0.0
    for i in range(len(pos)):
        d = np.linalg.norm(pos[i+1:] - pos[i], axis=1)
        if len(d):
            dmax = max(dmax, d.max())
    return float(dmax)


def com_offset(atoms):
    """Lateral (xy) offset between Hg COM and Au COM."""
    hg_com = np.mean([a.position for a in atoms if a.symbol == 'Hg'],
                     axis=0)
    au_com = np.mean([a.position for a in atoms if a.symbol == 'Au'],
                     axis=0)
    return hg_com[:2] - au_com[:2]


def top_layer_au(atoms, tol=0.5):
    au = [a for a in atoms if a.symbol == 'Au']
    z_top = max(a.position[2] for a in au)
    return np.array([a.position for a in au
                     if abs(a.position[2] - z_top) < tol])


def classify_site(atoms):
    """Nearest high-symmetry site to the Hg cluster COM."""
    hg_com = np.mean([a.position for a in atoms if a.symbol == 'Hg'],
                     axis=0)
    top = top_layer_au(atoms)
    if len(top) == 0:
        return "unknown", None

    dists = np.linalg.norm(top[:, :2] - hg_com[:2], axis=1)
    d = np.sort(dists)

    n_within_1_8 = int(np.sum(d < 1.8))
    n_within_3_4 = int(np.sum(d < 3.4))

    if n_within_1_8 == 3 and 4 <= n_within_3_4 <= 6:
        return "hollow (fcc/hcp)", d[:6]
    if n_within_1_8 == 2:
        return "bridge", d[:4]
    if n_within_1_8 == 1 and d[1] > 2.5:
        return "top", d[:4]
    return f"mixed/off-site ({n_within_1_8} Au within 1.8 Å)", d[:6]


def fcc_hollow_position(slab):
    """(x,y) of an fcc hollow site on the given slab (via ASE)."""
    probe = slab.copy()
    add_adsorbate(probe, Atom('He'), height=1.0, position='fcc')
    return probe.positions[-1][:2].copy()


# ============================================================
# Cell-size and vacuum checks
# ============================================================
def report_cell_checks(slab, hg_iso, label="start"):
    """Report cell-size and vacuum diagnostics."""
    cell = slab.cell
    a1 = np.linalg.norm(cell[0])
    a2 = np.linalg.norm(cell[1])
    a_avg = 0.5 * (a1 + a2)

    d_cluster = cluster_diameter(hg_iso, 'Hg')
    ratio = a_avg / d_cluster
    edge_gap = a_avg - d_cluster

    print(f"  --- Cell size checks ({label}) ---")
    print(f"  Cell side (avg in-plane):   {a_avg:.2f} Å")
    print(f"  Cluster diameter:           {d_cluster:.2f} Å")
    print(f"  Cell/cluster ratio:         {ratio:.2f}")
    print(f"  Edge-to-edge gap (isolated):{edge_gap:6.2f} Å")

    if ratio < IMAGE_WARN_RATIO:
        print(f"  WARNING: cell/cluster ratio < {IMAGE_WARN_RATIO:.1f}")
    else:
        print(f"  Cell is large enough.")

    # Vacuum
    z_slab = slab.positions[:, 2].max() - slab.positions[:, 2].min()
    z_cluster = hg_iso.positions[:, 2].max() - hg_iso.positions[:, 2].min()
    cell_z = np.linalg.norm(cell[2])
    eff_vac = cell_z - z_slab - z_cluster

    print(f"  --- Vacuum checks ({label}) ---")
    print(f"  Cell height (z):            {cell_z:.2f} Å")
    print(f"  Slab thickness:             {z_slab:.2f} Å")
    print(f"  Cluster thickness:          {z_cluster:.2f} Å")
    print(f"  Effective vacuum:           {eff_vac:.2f} Å")
    if eff_vac < 10.0:
        print(f"  WARNING: effective vacuum < 10 Å.")
    else:
        print(f"  Vacuum is sufficient.")
    return ratio, edge_gap, eff_vac


def report_post_relax_cell_checks(final):
    """Re-check image coupling using the *adsorbed* cluster dimensions."""
    cell = final.cell
    a_avg = 0.5 * (np.linalg.norm(cell[0]) + np.linalg.norm(cell[1]))
    d_ads = cluster_diameter(final, 'Hg')
    d_hg_hg = min_hg_hg_image_distance(final)
    edge_gap_ads = a_avg - d_ads

    print(f"  --- Post-relaxation image checks ---")
    print(f"  Cell side (avg in-plane):   {a_avg:.2f} Å")
    print(f"  Adsorbed cluster diameter:  {d_ads:.2f} Å")
    print(f"  Edge-to-edge gap (adsorbed):{edge_gap_ads:6.2f} Å")
    print(f"  Min Hg-Hg image distance:   {d_hg_hg:.3f} Å")

    if d_hg_hg < IMAGE_WARN_HG_HG:
        print(f"  WARNING: Hg-Hg image distance < "
              f"{IMAGE_WARN_HG_HG:.1f} Å — periodic images may interact.")
    else:
        print(f"  Hg-Hg image distance is large enough.")
    return a_avg, d_ads, d_hg_hg, edge_gap_ads


# ============================================================
# Relaxation driver
# ============================================================
def get_relaxed_energy(atoms, name, monitor_com=False,
                       poscar_out=None):
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
            print(f"  NOTE: cluster migrated > {DRIFT_WARN_THRESH} Å "
                  f"during relaxation.")

    if poscar_out is not None:
        write(poscar_out, atoms, format='vasp')
        print(f"  Saved final geometry to {poscar_out}")

    return E, fmax, dyn.nsteps


def save_trajectory_as_poscars(traj_file, prefix):
    """Write one POSCAR per frame of a .traj file."""
    try:
        frames = read(traj_file, index=':')
    except Exception as e:
        print(f"  Could not read {traj_file}: {e}")
        return
    for i, atoms in enumerate(frames):
        write(f'{prefix}_{i:04d}.vasp', atoms, format='vasp')
    print(f"  Saved {len(frames)} POSCAR frames as {prefix}_XXXX.vasp")


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
E_cluster, fmax_c, nsteps_c = get_relaxed_energy(
    hg13_iso, "cluster_iso", poscar_out="POSCAR_cluster_iso"
)
rg_iso = radius_of_gyration(hg13_iso, 'Hg')
d_iso  = cluster_diameter(hg13_iso, 'Hg')
print(f"  Cluster Rg (isolated):    {rg_iso:.3f} Å")
print(f"  Cluster diameter (iso):   {d_iso:.3f} Å")


# ============================================================
# 2. Clean Au(111) slab
# ============================================================
print("\n" + "=" * 68)
print(f"[2/3] Clean Au(111) slab {SLAB_SIZE} "
      f"({SLAB_SIZE[0]*SLAB_SIZE[1]*SLAB_SIZE[2]} Au atoms)")
print("=" * 68)
slab_clean = fcc111('Au', size=SLAB_SIZE, vacuum=VACUUM)
print(f"  Slab: {len(slab_clean)} Au atoms, no constraints")
E_slab, fmax_s, nsteps_s = get_relaxed_energy(
    slab_clean, "slab_clean", poscar_out="POSCAR_slab_clean"
)


# ============================================================
# 3. Combined system: Hg13 on Au(111)
# ============================================================
print("\n" + "=" * 68)
print(f"[3/3] Hg13 / Au(111) {SLAB_SIZE}  "
      f"(fcc placement, no constraints)")
print("=" * 68)

# Build fresh unconstrained slab
slab = fcc111('Au', size=SLAB_SIZE, vacuum=VACUUM)

# Build cluster, centered at origin
hg13 = Icosahedron('Hg', noshells=CLUSTER_SHELLS,
                   latticeconstant=HG_LATTICE)
hg13.center(vacuum=0)

# Cell-size and vacuum checks
report_cell_checks(slab, hg13, label="start")
print()

# Determine fcc hollow site
fcc_xy = fcc_hollow_position(slab)
au_com_xy = np.mean([a.position[:2] for a in slab if a.symbol == 'Au'],
                    axis=0)
print(f"  Slab COM (xy):    {au_com_xy}")
print(f"  fcc hollow (xy):  {fcc_xy}")

# Vertical placement
z_top         = slab.positions[:, 2].max()
z_cluster_min = hg13.positions[:, 2].min()
shift_z       = z_top + ADS_HEIGHT - z_cluster_min

# Lateral placement over fcc hollow
hg_com_xy = np.mean(hg13.positions[:, :2], axis=0)
shift_xy  = fcc_xy - hg_com_xy

hg13.translate([shift_xy[0], shift_xy[1], shift_z])

# Combine
system = slab + hg13

# Diagnostics BEFORE relaxation
dmin_start = min_hg_au_distance(system)
print(f"\n  Atoms: {len(system)} ({len(slab)} Au + {len(hg13)} Hg)")
print(f"  Min Hg-Au distance (start): {dmin_start:.3f} Å")
assert dmin_start > MIN_ALLOWED_HG_AU, (
    f"Atoms overlap at start: dmin = {dmin_start:.3f} Å"
)

site_start, _ = classify_site(system)
print(f"  Site at start (COM nearest to): {site_start}")

hg_com_placed = np.mean([a.position for a in system if a.symbol == 'Hg'],
                        axis=0)
d_fcc = np.linalg.norm(hg_com_placed[:2] - fcc_xy)
print(f"  Hg COM distance from fcc site: {d_fcc:.3f} Å")
assert d_fcc < 0.5, (
    f"Hg cluster is not placed over fcc: offset = {d_fcc:.3f} Å"
)

# Save initial combined structure
write("POSCAR_total_initial", system, format='vasp')
print(f"  Saved initial combined geometry to POSCAR_total_initial")

# Relax
E_total, fmax_t, nsteps_t = get_relaxed_energy(
    system, "total_system", monitor_com=True,
    poscar_out="POSCAR_total_final"
)


# ============================================================
# 4. Post-relaxation diagnostics
# ============================================================
print("\n" + "=" * 68)
print("Geometry diagnostics (final frame)")
print("=" * 68)

final = read('total_system.traj', index=-1)

dmin_end = min_hg_au_distance(final)
rg_ads   = radius_of_gyration(final, 'Hg')
d_ads    = cluster_diameter(final, 'Hg')

print(f"  Min Hg-Au distance (end):    {dmin_end:.3f} Å")
print(f"  Cluster Rg (adsorbed):       {rg_ads:.3f} Å")
print(f"  Cluster Rg (isolated):       {rg_iso:.3f} Å")
print(f"  Rg ratio (ads/iso):          {rg_ads / rg_iso:.3f}")
print(f"  Cluster diameter (ads):      {d_ads:.3f} Å")
print(f"  Cluster diameter (iso):      {d_iso:.3f} Å")
print(f"  Diameter ratio (ads/iso):    {d_ads / d_iso:.3f}")

# Periodic-image checks (using ADSORBED diameter)
print()
a_avg, d_ads_chk, d_hg_hg, edge_gap_ads = \
    report_post_relax_cell_checks(final)

# Registry check
site_end, dists_end = classify_site(final)
print(f"\n  Site at end (COM nearest to): {site_end}")
if dists_end is not None:
    print(f"    Nearest top-layer Au distances: "
          f"{[f'{d:.2f}' for d in dists_end]}")

hg_com_end = np.mean([a.position for a in final if a.symbol == 'Hg'],
                     axis=0)
d_fcc_end = np.linalg.norm(hg_com_end[:2] - fcc_xy)
print(f"  Hg COM distance from original fcc (end): {d_fcc_end:.3f} Å")
if d_fcc_end > DRIFT_WARN_THRESH:
    print(f"  NOTE: cluster migrated away from fcc by > "
          f"{DRIFT_WARN_THRESH} Å.")
    print(f"        Consider relabeling this result as '{site_end}'.")

# Contact / shape checks
print()
if dmin_end < MIN_ALLOWED_HG_AU:
    print("  WARNING: Hg and Au atoms are still overlapping.")
elif dmin_end > 4.0:
    print("  NOTE: Hg cluster may have desorbed (dmin > 4 Å).")
else:
    print("  Hg cluster is in contact with the surface.")

if rg_ads / rg_iso > 1.3:
    print("  WARNING: cluster has spread out (>30% Rg increase).")
elif rg_ads / rg_iso < 0.7:
    print("  WARNING: cluster has contracted significantly.")
else:
    print("  Cluster shape is preserved.")

# Save POSCAR frames of the combined-system trajectory
print()
save_trajectory_as_poscars("total_system.traj", "POSCAR_total_frame")


# ============================================================
# 5. Adsorption energy
# ============================================================
E_ads = E_total - (E_slab + E_cluster)
E_ads_per_atom = E_ads / n_hg

print("\n" + "=" * 68)
print("Result")
print("=" * 68)
print(f"  Slab:             {SLAB_SIZE} "
      f"({len(slab)} Au atoms)")
print(f"  Vacuum:           {VACUUM} Å")
print(f"  Cell side:        {a_avg:.2f} Å")
print(f"  Cell/cluster:     {a_avg / d_ads:.2f} "
      f"(using adsorbed diameter)")
print(f"  Min Hg-Hg image:  {d_hg_hg:.3f} Å")
print(f"  E_total          = {E_total:12.6f} eV")
print(f"  E_slab           = {E_slab:12.6f} eV")
print(f"  E_cluster        = {E_cluster:12.6f} eV")
print(f"  -----------------------------------------")
print(f"  E_ads (cluster)  = {E_ads:12.6f} eV")
print(f"  E_ads / Hg atom  = {E_ads_per_atom:12.6f} eV")
print(f"  Final site label = {site_end}")
print("=" * 68)

if E_total > 0:
    print("ERROR: E_total is positive — combined system is unphysical.")
elif E_ads > 0:
    print("The adsorption is ENDOTHERMIC (unstable).")
else:
    print("The adsorption is EXOTHERMIC (stable).")

all_ok = all(f < FMAX for f in (fmax_c, fmax_s, fmax_t))
print(f"\nConvergence: "
      f"{'all OK' if all_ok else 'WARNING — some runs did not converge'}")

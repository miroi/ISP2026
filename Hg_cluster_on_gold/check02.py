from ase.io import read
import numpy as np

def hg_offset(atoms):
    hg_com = np.mean([a.position for a in atoms if a.symbol == 'Hg'], axis=0)
    au_com = np.mean([a.position for a in atoms if a.symbol == 'Au'], axis=0)
    return hg_com[:2] - au_com[:2], np.linalg.norm(hg_com[:2] - au_com[:2])

traj = read('total_system.traj', index=':')
for i, atoms in enumerate(traj):
    if i % 10 == 0 or i == len(traj) - 1:
        vec, dist = hg_offset(atoms)
        print(f"Step {i:4d}: offset = {dist:.3f} Å  "
              f"(dx = {vec[0]:+.3f}, dy = {vec[1]:+.3f})")

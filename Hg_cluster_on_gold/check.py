from ase.io import read
import numpy as np

atoms = read('total_system.traj', index=0)
hg = [a for a in atoms if a.symbol == 'Hg']
au = [a for a in atoms if a.symbol == 'Au']

# Minimum Hg-Au distance
dmin = min(np.linalg.norm(h.position - a.position)
           for h in hg for a in au)
print(f"Min Hg-Au distance at start: {dmin:.3f} Å")

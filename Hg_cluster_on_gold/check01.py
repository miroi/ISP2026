from ase.io import read
import numpy as np

final = read('total_system.traj', index=-1)
hg_com = np.mean([a.position for a in final if a.symbol == 'Hg'], axis=0)
au_com = np.mean([a.position for a in final if a.symbol == 'Au'], axis=0)
print(f"Hg cluster lateral offset from Au slab center: "
      f"{np.linalg.norm(hg_com[:2] - au_com[:2]):.3f} Å")

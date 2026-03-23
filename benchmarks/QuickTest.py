
import sys
import os
# Allow importing modules from the root repository
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from SEDE import spatial_entropy_differential_evolution

def sphere(x): return np.sum(np.square(x))

dims = [10, 100, 500]
for d in dims:
    print(f"Running SEDE Quick Check (Sphere {d}D)...")
    _, best_y, _ = spatial_entropy_differential_evolution(
        sphere, dim=d, pop_size=50, max_iter=1500, lb=-100, ub=100
    )
    print(f"Final Score ({d}D): {best_y}\n")
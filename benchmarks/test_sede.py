
import sys
import os
# Allow importing modules from the root repository
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from SEDE import spatial_entropy_differential_evolution
def sphere(x): return np.sum(np.square(x))
_, best_y, history = spatial_entropy_differential_evolution(sphere, dim=30, pop_size=50, max_iter=1500)
print("Final Y:", best_y)

import numpy as np
import time
from SEDE import spatial_entropy_differential_evolution
import SEDE
import sede_core

def sphere(x): return np.sum(np.square(x))

def main():
    print("Running SEDE Python (Numba) vs SEDE C++ (OpenMP)...")
    dim = 100
    pop_size = 50
    max_iter = 1000
    
    # 1. SEDE Python (Numba)
    print("1. Starting SEDE Python (Numba)...")
    start_py = time.time()
    _, score_py, _ = spatial_entropy_differential_evolution(
        sphere, dim=dim, pop_size=pop_size, max_iter=max_iter, lb=-100, ub=100, seed=42
    )
    time_py = time.time() - start_py
    print(f"   Python Run Time: {time_py:.2f}s | Score: {score_py:.2e}\n")
    
    # Python Monkey-Patch SEDE.compute_spatial_entropy_numba to call sede_core wrapper
    def proxy_cpp(X, N):
        X_contig = np.ascontiguousarray(X, dtype=np.float64)
        return sede_core.compute_spatial_entropy_cpp(X_contig)
    
    SEDE.compute_spatial_entropy_numba = proxy_cpp

    # 2. SEDE C++ (OpenMP)
    print("2. Starting SEDE C++ (OpenMP)...")
    start_cpp = time.time()
    _, score_cpp, _ = spatial_entropy_differential_evolution(
        sphere, dim=dim, pop_size=pop_size, max_iter=max_iter, lb=-100, ub=100, seed=42
    )
    time_cpp = time.time() - start_cpp
    print(f"   C++ Run Time: {time_cpp:.2f}s | Score: {score_cpp:.2e}\n")
    
    speedup = time_py / time_cpp
    print(f"🔥 SPEEDUP FACTOR: {speedup:.2f}x faster using C++ OpenMP!")

if __name__ == '__main__':
    main()

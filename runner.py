import argparse
import numpy as np
import time

from SEDE import spatial_entropy_differential_evolution
import SEDE
import sede_core

def sphere(x): return np.sum(np.square(x))
def rosenbrock(x): return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
def rastrigin(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Map Python Numba Logic strictly to C++ Core Backend 
def proxy_cpp(X, N):
    X_contig = np.ascontiguousarray(X, dtype=np.float64)
    return sede_core.compute_spatial_entropy_cpp(X_contig)
    
SEDE.compute_spatial_entropy_numba = proxy_cpp

def main():
    parser = argparse.ArgumentParser(description="SEDE Optimizer CLI (C++ Accelerated)")
    parser.add_argument('--func', type=str, default='sphere', choices=['sphere', 'rosenbrock', 'rastrigin'], help='Benchmark function')
    parser.add_argument('--dim', type=int, default=100, help='Dimensionality')
    parser.add_argument('--iter', type=int, default=1000, help='Max Iterations')
    parser.add_argument('--pop', type=int, default=50, help='Population Size')
    
    args = parser.parse_args()
    
    print(f"--- SEDE C++ Optimizer ---")
    print(f"Function: {args.func.upper()} | Dim: {args.dim} | Pop: {args.pop} | Iter: {args.iter}")
    
    func_map = {
        'sphere': (sphere, -100, 100),
        'rosenbrock': (rosenbrock, -30, 30),
        'rastrigin': (rastrigin, -5.12, 5.12)
    }
    
    func, lb, ub = func_map[args.func.lower()]
    
    start_time = time.time()
    best_x, best_y, _ = spatial_entropy_differential_evolution(
        func, dim=args.dim, pop_size=args.pop, max_iter=args.iter, lb=lb, ub=ub, seed=None
    )
    time_taken = time.time() - start_time
    
    print(f"\n[RESULTS]")
    print(f"Time Taken: {time_taken:.3f}s")
    print(f"Best Score: {best_y:.3e}")

if __name__ == '__main__':
    main()

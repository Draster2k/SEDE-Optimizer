import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Import your algorithms
from SEDE import spatial_entropy_differential_evolution
from sko.PSO import PSO
from sko.DE import DE
from sko.GA import GA

# === Benchmark Functions ===
def sphere(x): return np.sum(np.square(x))
def rosenbrock(x): return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
def rastrigin(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def ackley(x):
    d = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2)/d)) - np.exp(np.sum(np.cos(2*np.pi*x))/d) + 20 + np.e

# Literature Standard Bounds for Benchmarks
benchmarks = {
    "Sphere": (sphere, -100, 100),
    "Rosenbrock": (rosenbrock, -30, 30),
    "Rastrigin": (rastrigin, -5.12, 5.12),
    "Ackley": (ackley, -32, 32)
}

# === Experiment Settings ===
DIMS = [10, 50, 100]           
RUNS = 30             
POP_SIZE = 50         
MAX_ITER = 1500       

# === Worker Function ===
def run_single_trial(run_id, func, dim, lb, ub):
    seed = 42 + run_id
    # CRITICAL: Force the global numpy RNG to respect the seed for 'sko' algorithms
    np.random.seed(seed) 
    
    results = {}
    
    # 1. SEDE
    start = time.time()
    _, best_y, _ = spatial_entropy_differential_evolution(func, dim, POP_SIZE, MAX_ITER, lb, ub, seed=seed)
    results['SEDE'] = float(best_y)
    results['SEDE_time'] = time.time() - start
    
    # 2. PSO
    start = time.time()
    pso = PSO(func=func, dim=dim, pop=POP_SIZE, max_iter=MAX_ITER, lb=[lb]*dim, ub=[ub]*dim)
    pso.run()
    # CRITICAL: Extract as float to prevent Pandas array-column crashes
    results['PSO'] = float(np.min(pso.gbest_y)) 
    results['PSO_time'] = time.time() - start
    
    # 3. DE
    start = time.time()
    de = DE(func=func, n_dim=dim, size_pop=POP_SIZE, max_iter=MAX_ITER, lb=[lb]*dim, ub=[ub]*dim)
    _, best_y_de = de.run()
    results['DE'] = float(np.min(best_y_de))
    results['DE_time'] = time.time() - start
    
    # 4. GA
    start = time.time()
    ga = GA(func=func, n_dim=dim, size_pop=POP_SIZE, max_iter=MAX_ITER, lb=[lb]*dim, ub=[ub]*dim, precision=1e-7)
    _, best_y_ga = ga.run()
    results['GA'] = float(np.min(best_y_ga))
    results['GA_time'] = time.time() - start

    return results

# === Main Execution ===
if __name__ == "__main__":
    os.makedirs("Results", exist_ok=True)
    
    print(f"Starting Paper Benchmark (30D, 1500 Iterations)...")
    
    for func_name, (func, lb, ub) in benchmarks.items():
        for dim in DIMS:
            print(f"Running {func_name} - {dim}D [Bounds: {lb} to {ub}]...")
            
            # Run parallel trials
            trial_results = Parallel(n_jobs=6)(
                delayed(run_single_trial)(run, func, dim, lb, ub) for run in range(RUNS)
            )
            
            df = pd.DataFrame(trial_results)
            
            # Reorder columns safely
            output_cols = ['SEDE', 'PSO', 'DE', 'GA', 'SEDE_time', 'PSO_time', 'DE_time', 'GA_time']
            df = df[output_cols]
            
            filename = f"Results/{func_name}_{dim}D.csv"
            df.to_csv(filename, index=False)
            
            sede_mean = df['SEDE'].mean()
            print(f"  -> Saved {filename} | SEDE Mean: {sede_mean:.2e}")

    print("\n✅ Benchmark Complete. Run 'python Analyze_data.py' to get the clean table.")
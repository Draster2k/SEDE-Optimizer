import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import os

from SEDE import spatial_entropy_differential_evolution
from sko.DE import DE
from sko.PSO import PSO

# Load Dataset
data = load_breast_cancer()
X_data = StandardScaler().fit_transform(data.data)
y_data = data.target

kernels = ['linear', 'poly', 'rbf']

def fitness_function(x):
    # Decode parameters from [0, 1] bounds
    
    # C in log-space [0.1, 1000]
    C = 10 ** (-1 + x[0] * 4) 
    
    # Gamma in log-space [0.0001, 1]
    gamma = 10 ** (-4 + x[1] * 4)
    
    # Kernel index mapping
    k_idx = int(np.clip(np.floor(x[2] * 3), 0, 2))
    kernel = kernels[k_idx]
    
    # Build SVM classifier
    clf = SVC(C=C, kernel=kernel, gamma=gamma, cache_size=500, random_state=42)
    
    # 5-fold CV (n_jobs=1 prevents parallel-spawn bottlenecking on small data)
    scores = cross_val_score(clf, X_data, y_data, cv=5, n_jobs=1)
    acc = scores.mean()
    
    return 1.0 - acc

def main():
    dim = 3
    pop_size = 20
    max_iter = 30
    
    # Using continuous bounds [0,1] mapping internally inside fitness_func
    lb = 0.0 
    ub = 1.0 
    
    bounds_lb = [0.0] * dim
    bounds_ub = [1.0] * dim

    print("Running Real-World HPO SVM Benchmark...")
    print("Dataset: Breast Cancer (30 features, 569 samples)")
    print(f"Iters: {max_iter}, Pop: {pop_size}\n")
    
    # 1. SEDE
    print("Starting SEDE...")
    start = time.time()
    _, best_y_sede, history_sede = spatial_entropy_differential_evolution(
        fitness_function, dim=dim, pop_size=pop_size, max_iter=max_iter, lb=lb, ub=ub, seed=42
    )
    t_sede = time.time() - start
    sede_acc = 1.0 - best_y_sede
    print(f"SEDE Acc: {sede_acc*100:.2f}% | Time: {t_sede:.1f}s")
    
    # 2. DE
    print("Starting DE...")
    start = time.time()
    de = DE(func=fitness_function, n_dim=dim, size_pop=pop_size, max_iter=max_iter, lb=bounds_lb, ub=bounds_ub)
    _, best_y_de = de.run()
    t_de = time.time() - start
    de_acc = 1.0 - np.min(best_y_de)
    history_de = [np.min(de.generation_best_Y[:i+1]) for i in range(max_iter)]
    print(f"DE Acc: {de_acc*100:.2f}% | Time: {t_de:.1f}s")
    
    # 3. PSO
    print("Starting PSO...")
    start = time.time()
    pso = PSO(func=fitness_function, dim=dim, pop=pop_size, max_iter=max_iter, lb=bounds_lb, ub=bounds_ub)
    pso.run()
    t_pso = time.time() - start
    pso_acc = 1.0 - np.min(pso.gbest_y)
    
    # Format PSO history correctly to prevent array mismatches
    history_pso = pso.gbest_y_hist if hasattr(pso, 'gbest_y_hist') else [np.min(pso.gbest_y)] * max_iter
    if len(history_pso) < max_iter:
        history_pso = list(history_pso) + [history_pso[-1]] * (max_iter - len(history_pso))
        
    print(f"PSO Acc: {pso_acc*100:.2f}% | Time: {t_pso:.1f}s\n")
    
    # Plot Convergence Curve (Accuracy)
    plt.figure(figsize=(10, 6))
    
    # Strip generation 0 from SEDE to match exact iteration count against DE
    acc_sede = [(1.0 - y) * 100 for y in history_sede[1:]]
    acc_de = [(1.0 - y) * 100 for y in history_de]
    acc_pso = [(1.0 - y) * 100 for y in history_pso[:max_iter]]
    
    iters = np.arange(1, max_iter + 1)
    
    plt.plot(iters, acc_sede, label=f'SEDE (Ours) : {sede_acc*100:.2f}%', linewidth=3, color='crimson', marker='*')
    plt.plot(iters, acc_de, label=f'Standard DE : {de_acc*100:.2f}%', linewidth=2, color='forestgreen', marker='s')
    plt.plot(iters, acc_pso, label=f'Standard PSO : {pso_acc*100:.2f}%', linewidth=2, color='royalblue', marker='o')
    
    plt.title("Convergence Curve: SVM Hyperparameter Optimization", fontsize=16, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Validation Accuracy (%)", fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs("Results", exist_ok=True)
    save_path = "Results/RealWorld_Conv.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Generated Convergence Plot: {save_path}")

if __name__ == "__main__":
    main()

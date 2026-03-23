
import sys
import os
# Allow importing modules from the root repository
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

from SEDE import spatial_entropy_differential_evolution
from sko.PSO import PSO
from sko.DE import DE

def sphere(x): return np.sum(np.square(x))
def rosenbrock(x): return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
def rastrigin(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

benchmarks = {
    'Sphere': (sphere, -100, 100),
    'Rosenbrock': (rosenbrock, -30, 30),
    'Rastrigin': (rastrigin, -5.12, 5.12)
}

DIMS = [10, 50, 100]
RUNS = 30
POP_SIZE = 50
MAX_ITER = 1500

def run_single(func, dim, lb, ub, algo, seed):
    np.random.seed(seed)
    if algo == 'SEDE':
        b_x, b_y, hist = spatial_entropy_differential_evolution(
            func, dim, POP_SIZE, MAX_ITER, lb, ub, seed=seed
        )
        return float(b_y), [float(h) for h in hist[1:]]
    elif algo == 'PSO':
        pso = PSO(func=func, dim=dim, pop=POP_SIZE, max_iter=MAX_ITER, lb=[lb]*dim, ub=[ub]*dim)
        pso.run()
        b_y = np.min(pso.gbest_y)
        hist = pso.gbest_y_hist if hasattr(pso, 'gbest_y_hist') else [b_y]*MAX_ITER
        return float(b_y), [float(h) for h in hist]
    elif algo == 'DE':
        de = DE(func=func, n_dim=dim, size_pop=POP_SIZE, max_iter=MAX_ITER, lb=[lb]*dim, ub=[ub]*dim)
        de.run()
        b_y = np.min(de.generation_best_Y)
        hist = [float(np.min(de.generation_best_Y[:i+1])) for i in range(MAX_ITER)]
        return float(b_y), hist

def main():
    os.makedirs('Results', exist_ok=True)
    results_list = []
    histories = {}
    
    print("Beginning Deep Benchmarks (This may take several minutes)...")
    for name, (func, lb, ub) in benchmarks.items():
        histories[name] = {}
        for d in DIMS:
            histories[name][str(d)] = {}
            print(f"-> Benchmarking {name} {d}D")
            for algo in ['SEDE', 'DE', 'PSO']:
                t0 = time.time()
                # Deploying to parallel cores
                out = Parallel(n_jobs=-1)(delayed(run_single)(func, d, lb, ub, algo, s) for s in range(RUNS))
                
                scores = [o[0] for o in out]
                best_idx = np.argmin(scores)
                
                # Make history length max_iter perfectly
                hist_trunc = out[best_idx][1][:MAX_ITER]
                while len(hist_trunc) < MAX_ITER: hist_trunc.append(hist_trunc[-1])
                
                histories[name][str(d)][algo] = hist_trunc
                
                for r_idx, s in enumerate(scores):
                    results_list.append({
                        'Function': name,
                        'Dimension': d,
                        'Algorithm': algo,
                        'Run': r_idx + 1,
                        'Score': s
                    })
                print(f"   {algo} Average: {np.mean(scores):.2e} | Time/Cluster: {time.time()-t0:.1f}s")
                
    df = pd.DataFrame(results_list)
    df.to_csv('Results/Final_Baselines.csv', index=False)
    
    # Write JSON metadata (Wilcoxon)
    p_values = {}
    df_pivot = df.pivot_table(index=['Function', 'Dimension', 'Run'], columns='Algorithm', values='Score').reset_index()
    for (f, d), grp in df_pivot.groupby(['Function', 'Dimension']):
        key = f"{f}_{d}D"
        sede, de, pso = grp['SEDE'], grp['DE'], grp['PSO']
        try: p_de = wilcoxon(sede, de)[1]
        except: p_de = 1.0
        try: p_pso = wilcoxon(sede, pso)[1]
        except: p_pso = 1.0
        p_values[key] = {'SEDE_vs_DE': p_de, 'SEDE_vs_PSO': p_pso}
        
    with open('Results/research_log.json', 'w') as f:
        json.dump(p_values, f, indent=4)
        print("✅ Exported research_log.json")
        
    # Generate Plots
    print("Generating High-Res Plots...")
    
    # 1. Boxplot
    plt.figure(figsize=(12, 8))
    df['Score_Plot'] = np.maximum(df['Score'], 1e-250)
    sns.boxplot(data=df, x='Function', y='Score_Plot', hue='Algorithm')
    plt.yscale('log')
    plt.title('Spread of Final Scores (Lower is Better)')
    plt.savefig('Results/Boxplot_Accuracy.png', dpi=300)
    plt.close()
    print("✅ Exported Boxplot_Accuracy.png")
    
    # 2. Convergence
    plt.figure(figsize=(15, 6))
    for i, func in enumerate(['Sphere', 'Rosenbrock', 'Rastrigin'], 1):
        plt.subplot(1, 3, i)
        for algo in ['SEDE', 'DE', 'PSO']:
            y10 = np.log10(np.maximum(histories[func]['10'][algo], 1e-250))
            y100 = np.log10(np.maximum(histories[func]['100'][algo], 1e-250))
            plt.plot(y10, label=f'{algo} 10D', linestyle='solid')
            plt.plot(y100, label=f'{algo} 100D', linestyle='dashed')
        plt.title(f'{func}: Convergence')
        plt.xlabel('Iterations')
        plt.ylabel('Log10(Score)')
        if i==1: plt.legend(prop={'size': 8})
    plt.tight_layout()
    plt.savefig('Results/Convergence_10D_30D.png', dpi=300)
    plt.close()
    print("✅ Exported Convergence_10D_30D.png")
    
    # 3. Heatmap
    rank_matrix = []
    labels = []
    for (f, d), grp in df_pivot.groupby(['Function', 'Dimension']):
        ranks = grp[['DE', 'PSO', 'SEDE']].rank(axis=1).mean()
        rank_matrix.append(ranks.values)
        labels.append(f"{f} {d}D")
        
    plt.figure(figsize=(6, 5))
    sns.heatmap(rank_matrix, annot=True, xticklabels=['DE', 'PSO', 'SEDE'], yticklabels=labels, cmap='viridis_r')
    plt.title('Average Rank per Configuration')
    plt.tight_layout()
    plt.savefig('Results/Rank_Heatmap.png', dpi=300)
    plt.close()
    print("✅ Exported Rank_Heatmap.png")
    
    # Markdown Table creation
    md = "### Final Python Metrics: SEDE vs Baselines (30 Runs)\n\n"
    md += "| Function | Dim | SEDE Mean | DE Mean | PSO Mean | SEDE vs DE (p) | SEDE vs PSO (p) |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for (f, d), grp in df_pivot.groupby(['Function', 'Dimension']):
        s_m, de_m, pso_m = grp['SEDE'].mean(), grp['DE'].mean(), grp['PSO'].mean()
        p1, p2 = p_values[f"{f}_{d}D"]['SEDE_vs_DE'], p_values[f"{f}_{d}D"]['SEDE_vs_PSO']
        md += f"| {f} | {d} | {s_m:.2e} | {de_m:.2e} | {pso_m:.2e} | {p1:.3f} | {p2:.3f} |\n"
        
    with open('Results/Final_Metrics_Table.md', 'w') as f:
        f.write(md)
        
    print("✅ All Final Python Operations Completed successfully.")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()

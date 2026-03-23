import re
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. TEX Parsing ---
def parse_tex_value(val_str):
    """Extracts the mean floating point value from a LaTeX table string."""
    val_str = re.sub(r'\\textbf{(.*?)}', r'\1', val_str)
    mean_str = val_str.split(r'\pm')[0].strip()
    mean_str = re.sub(r'\\times 10\^{([^}]+)}', r'e\1', mean_str)
    mean_str = mean_str.replace('$', '')
    try:
        return float(mean_str)
    except:
        return np.nan

def load_results(filepath):
    """Parses results_table.tex into a dictionary of function -> {algo: score}"""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    algos = []
    results = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith('Function &'):
            # Parse header
            parts = line.split('&')
            algos = [p.strip().replace('\\\\', '').strip() for p in parts[1:]]
        elif '30D &' in line:
            # Parse row
            parts = line.split('&')
            func_name = parts[0].replace('30D', '').strip()
            scores = {}
            for i, algo in enumerate(algos):
                scores[algo] = parse_tex_value(parts[i+1])
            results[func_name] = scores
            
    return algos, results

# --- 2. 2D Benchmark Functions ---
def sphere_2d(X, Y): 
    return X**2 + Y**2

def rosenbrock_2d(X, Y): 
    return 100 * (Y - X**2)**2 + (1 - X)**2

def rastrigin_2d(X, Y): 
    return 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

def ackley_2d(X, Y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5*(X**2 + Y**2))) - np.exp(0.5*(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))) + 20 + np.e

benchmarks = {
    'Sphere': (sphere_2d, -100, 100, 0, 0),
    'Rosenbrock': (rosenbrock_2d, -2, 2, 1, 1),
    'Rastrigin': (rastrigin_2d, -5.12, 5.12, 0, 0),
    'Ackley': (ackley_2d, -32, 32, 0, 0)
}

# --- 3. Visualization ---
def plot_3d_results():
    filepath = "results_table.tex"
    algos, results = load_results(filepath)
    if not results: return
    
    colors = {'SEDE': 'red', 'PSO': 'blue', 'DE': 'green', 'GA': 'purple'}
    markers = {'SEDE': '*', 'PSO': 'o', 'DE': 's', 'GA': '^'}
    
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Optimization Results on 2D Function Surfaces', fontsize=20, fontweight='bold')
    
    for idx, (func_name, (func, lb, ub, x_opt, y_opt)) in enumerate(benchmarks.items(), 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # 1. Plot continuous surface
        # Narrow bounds for visual clarity near the minimum
        span = (ub - lb) * 0.1 
        X = np.linspace(x_opt - span, x_opt + span, 100)
        Y = np.linspace(y_opt - span, y_opt + span, 100)
        X, Y = np.meshgrid(X, Y)
        Z = func(X, Y)
        
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')
        
        # 2. Plot result points
        if func_name in results:
            scores = results[func_name]
            # Offset x, y slightly so markers don't overlap entirely
            offsets = [(0,0), (0.02*span, 0), (-0.02*span, 0), (0, 0.02*span)]
            
            for i, algo in enumerate(algos):
                z_val = scores.get(algo, np.nan)
                if not pd.isna(z_val):
                    ox, oy = offsets[i % len(offsets)]
                    ax.scatter(x_opt + ox, y_opt + oy, z_val, 
                               color=colors.get(algo, 'black'), 
                               marker=markers.get(algo, 'o'), 
                               s=150, label=f'{algo} ({z_val:.1e})',
                               depthshade=False, edgecolors='black', linewidth=1)
                    
                    # Add a drop line to the surface
                    ax.plot([x_opt + ox, x_opt + ox], [y_opt + oy, y_opt + oy], 
                            [func(x_opt+ox, y_opt+oy), z_val], 
                            color=colors.get(algo, 'black'), linestyle='--', alpha=0.5)
        
        ax.set_title(f'{func_name}', fontsize=16)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Fitness (Z)')
        if idx == 1: # only one legend needed
            ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs('Results', exist_ok=True)
    plt.savefig('Results/3D_Results_Visualization.png', dpi=300)
    print("✅ Successfully generated 3D visualization: Results/3D_Results_Visualization.png")

if __name__ == "__main__":
    import pandas as pd # required for pd.isna
    plot_3d_results()

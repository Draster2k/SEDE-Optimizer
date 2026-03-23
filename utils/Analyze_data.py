
import sys
import os
# Allow importing modules from the root repository
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
import warnings
import re

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Results"))
ALGORITHMS = ["SEDE", "PSO", "DE", "GA"] 
LATEX_RESULTS_FILE = "results_table.tex"
LATEX_RANKS_FILE = "ranks_table.tex"

def format_sci_latex(n):
    if pd.isna(n): return "-"
    if n == 0: return "0"
    if 1e-3 < abs(n) < 1e3:
        return f"{n:.2f}"
    a, b = "{:.2e}".format(n).split("e")
    b = int(b)
    return f"${a} \\times 10^{{{b}}}$"

def extract_dim(filename):
    """Extracts the numerical dimension from the filename string."""
    match = re.search(r"(\d+)D", filename)
    return int(match.group(1)) if match else 0

def load_and_aggregate():
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    if not csv_files:
        print("❌ No CSV files found!")
        return None

    summary_rows = []
    print(f"📂 Found {len(csv_files)} result files. Processing...")

    for file in sorted(csv_files):
        filename = os.path.basename(file).replace(".csv", "")
        dim = extract_dim(filename)
        func_name = filename.split("_")[0]
        
        try:
            df = pd.read_csv(file)
            for algo in ALGORITHMS:
                if algo in df.columns:
                    df[algo] = pd.to_numeric(df[algo], errors='coerce')

            row = {
                "Benchmark": func_name,
                "Dim": dim,
                "Display": f"{func_name} {dim}D"
            }
            
            for algo in ALGORITHMS:
                data = df[algo].dropna()
                row[f"{algo}_mean"] = data.mean() if len(data) > 0 else np.nan
                row[f"{algo}_std"] = data.std() if len(data) > 0 else np.nan
            
            summary_rows.append(row)
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")

    summary_df = pd.DataFrame(summary_rows)
    # Sort by Function Name, then by Dimension
    summary_df = summary_df.sort_values(by=["Benchmark", "Dim"])
    return summary_df

def generate_latex_table(summary_df):
    with open(LATEX_RESULTS_FILE, "w") as f:
        f.write("\\begin{table*}[htbp]\n\\centering\n")
        f.write("\\caption{Benchmark Results across 10D, 50D, and 100D. Best results are bolded.}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        
        col_setup = "l" + "c" * len(ALGORITHMS)
        f.write(f"\\begin{{tabular}}{{{col_setup}}}\n\\hline\n")
        f.write("Function & " + " & ".join(ALGORITHMS) + " \\\\\n\\hline\n")

        for _, row in summary_df.iterrows():
            line_items = [row["Display"]]
            means = [row[f"{algo}_mean"] for algo in ALGORITHMS]
            best_mean = np.nanmin(means)

            for algo in ALGORITHMS:
                m = row[f"{algo}_mean"]
                s = row[f"{algo}_std"]
                txt = f"{format_sci_latex(m)} $\\pm$ {format_sci_latex(s)}"
                
                # PRECISION BOLDING FIX:
                # Bolds if exactly equal or within a tiny relative tolerance
                if not pd.isna(m):
                    is_best = False
                    if m == best_mean:
                        is_best = True
                    elif best_mean != 0 and abs(m - best_mean) / abs(best_mean) < 1e-8:
                        is_best = True
                    
                    if is_best:
                        txt = f"\\textbf{{{txt}}}"
                
                line_items.append(txt)
            f.write(" & ".join(line_items) + " \\\\\n")

        f.write("\\hline\n\\end{tabular}\n}\n\\end{table*}\n")
    print(f"✅ Saved Multi-Dim LaTeX Table to: {LATEX_RESULTS_FILE}")

def run_statistical_tests(summary_df):
    print("\n📊 Running Statistical Tests...")
    data_matrix = np.array([[row[f"{algo}_mean"] for algo in ALGORITHMS] for _, row in summary_df.iterrows()])
    
    if np.isnan(data_matrix).any():
        data_matrix = np.nan_to_num(data_matrix, nan=np.inf)

    stat, p_value = friedmanchisquare(*data_matrix.T)
    print(f"   -> Friedman Test: Chi2={stat:.3f}, p-value={p_value:.3e}")
    
    ranks = np.array([rankdata(row) for row in data_matrix])
    avg_ranks = np.mean(ranks, axis=0)
    
    with open(LATEX_RANKS_FILE, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\begin{tabular}{lc}\n\\hline\nAlgorithm & Avg Rank \\\\\n\\hline\n")
        for i in np.argsort(avg_ranks):
            f.write(f"{ALGORITHMS[i]} & {avg_ranks[i]:.2f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
        
    print(f"✅ Saved Ranks Table to: {LATEX_RANKS_FILE}")

    # Wilcoxon: SEDE vs Runner-up
    try:
        sede_idx = ALGORITHMS.index("SEDE")
        sorted_indices = np.argsort(avg_ranks)
        runner_up_idx = sorted_indices[0] if sorted_indices[0] != sede_idx else sorted_indices[1]
        
        sede_scores = data_matrix[:, sede_idx]
        runner_up_scores = data_matrix[:, runner_up_idx]
        
        if not np.allclose(sede_scores, runner_up_scores):
            _, p_val_w = wilcoxon(sede_scores, runner_up_scores)
            print(f"   -> Wilcoxon (SEDE vs {ALGORITHMS[runner_up_idx]}): p={p_val_w:.3e}")
    except Exception as e:
        print(f"   ❌ Wilcoxon Error: {e}")

if __name__ == "__main__":
    df = load_and_aggregate()
    if df is not None:
        generate_latex_table(df)
        run_statistical_tests(df)
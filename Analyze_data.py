import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import matplotlib.pyplot as plt

# === Algorithms ===
algorithms = ["EGO", "DE", "GA", "PSO"]

# === Load results ===
def load_results(results_dir="Results"):
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    results = {}
    for file in files:
        name = os.path.basename(file).replace(".csv", "")
        df = pd.read_csv(file)
        results[name] = df
    return results

# === Friedman Test ===
def friedman_test(results):
    data = []
    for name, df in results.items():
        values = [df[algo].mean() for algo in algorithms]
        data.append(values)
    data = np.array(data).T
    stat, p = friedmanchisquare(*data)
    return stat, p

# === Average Ranks ===
def compute_ranks(results):
    ranks = {algo: [] for algo in algorithms}
    for _, df in results.items():
        means = {algo: df[algo].mean() for algo in algorithms}
        sorted_algos = sorted(means.items(), key=lambda x: x[1])
        for rank, (algo, _) in enumerate(sorted_algos, start=1):
            ranks[algo].append(rank)
    avg_ranks = {algo: np.mean(r) for algo, r in ranks.items()}
    return avg_ranks

# === Critical Difference Diagram ===
def cd_diagram(avg_ranks, save_path="Results/cd_diagram.png"):
    algos, ranks = zip(*sorted(avg_ranks.items(), key=lambda x: x[1]))
    plt.figure(figsize=(8, 2))
    plt.hlines(1, min(ranks) - 0.5, max(ranks) + 0.5, color="k")
    for i, (algo, rank) in enumerate(zip(algos, ranks)):
        plt.plot(rank, 1, "o", label=algo)
        plt.text(rank, 1.05, algo, ha="center")
    plt.xlabel("Average Rank (lower is better)")
    plt.yticks([])
    plt.legend().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === LaTeX Table: Results per Benchmark ===
def save_latex_results(results, save_path="Results/results_table.tex"):
    with open(save_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Function & EGO & DE & GA & PSO \\\\\n")
        f.write("\\hline\n")
        for name, df in results.items():
            row = [f"{df[algo].mean():.2e} $\\pm$ {df[algo].std():.1e}" for algo in algorithms]
            f.write(f"{name} & " + " & ".join(row) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Benchmark results (mean $\\pm$ std over 30 runs).}\n")
        f.write("\\end{table}\n")

# === LaTeX Table: Ranks ===
def save_latex_ranks(avg_ranks, save_path="Results/ranks_table.tex"):
    with open(save_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\hline\n")
        f.write("Algorithm & Average Rank \\\\\n")
        f.write("\\hline\n")
        for algo, rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
            f.write(f"{algo} & {rank:.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Average ranks across benchmarks (lower is better).}\n")
        f.write("\\end{table}\n")

# === Main ===
def main():
    results = load_results("Results")

    # Friedman test
    stat, p = friedman_test(results)
    print("\n=== Friedman Test ===")
    print(f"Chi2 = {stat:.3f}, p = {p:.3e}")

    # Average ranks
    avg_ranks = compute_ranks(results)
    print("\n=== Average Ranks ===")
    for algo, rank in avg_ranks.items():
        print(f"{algo}: {rank:.2f}")

    # CD diagram
    cd_diagram(avg_ranks)

    # Save LaTeX tables
    save_latex_results(results)
    save_latex_ranks(avg_ranks)
    print("\nLaTeX tables saved to Results/")

if __name__ == "__main__":
    main()
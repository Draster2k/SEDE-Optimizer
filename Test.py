import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from EGO import entropy_guided_optimization
from sko.PSO import PSO
from sko.DE import DE

# === Benchmark Functions ===
def sphere(x): return np.sum(x ** 2)
def rosenbrock(x): return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
def rastrigin(x): return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
def ackley(x):
    d = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d)) \
           - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + np.e
def griewank(x):
    d = len(x)
    return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
def schwefel(x): return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
def zakharov(x):
    i = np.arange(1, len(x) + 1)
    return np.sum(x ** 2) + (np.sum(0.5 * i * x)) ** 2 + (np.sum(0.5 * i * x)) ** 4
def michalewicz(x, m=10):
    i = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) * (np.sin(i * x ** 2 / np.pi)) ** (2 * m))
def elliptic(x):
    d = len(x)
    exps = np.linspace(0, 6, d)
    coeffs = 10 ** exps
    return np.sum(coeffs * (x ** 2))
def step(x): return np.sum(np.floor(x + 0.5) ** 2)
def noisy_quartic(x):
    i = np.arange(1, len(x) + 1)
    return np.sum(i * (x ** 4)) + np.random.rand()
def levy(x):
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[0]) ** 2 + np.sum((w[:-1] - 1) ** 2 *
           (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)) + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
def schaffer_f6(x):
    x1, x2 = x[0], x[1]
    num = np.sin(np.sqrt(x1 ** 2 + x2 ** 2)) ** 2 - 0.5
    den = (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2
    return 0.5 + num / den
def fletcher_powell(x):
    d = len(x)
    i = np.arange(1, d + 1)
    return np.sum(i * (x ** 2)) + np.sum(np.sin(x) ** 2)

benchmarks = {
    "Sphere": sphere, "Rosenbrock": rosenbrock, "Rastrigin": rastrigin,
    "Ackley": ackley, "Griewank": griewank, "Schwefel": schwefel,
    "Zakharov": zakharov, "Michalewicz": michalewicz,
    "Elliptic": elliptic, "Step": step, "NoisyQuartic": noisy_quartic,
    "Levy": levy, "SchafferF6": schaffer_f6, "FletcherPowell": fletcher_powell,
}

# === Wrapper for scikit-opt ===
def wrap(func):
    def f(X):
        X = np.atleast_2d(X)
        return np.array([func(x) for x in X])
    return f

def run_pso(func, dim, max_iter, pop_size, lb, ub):
    f = wrap(func)
    pso = PSO(func=f, dim=dim, pop=pop_size, max_iter=max_iter, lb=[lb]*dim, ub=[ub]*dim)
    pso.run()
    return float(np.squeeze(pso.gbest_y))

def run_de(func, dim, max_iter, pop_size, lb, ub):
    f = wrap(func)
    de = DE(func=f, n_dim=dim, size_pop=pop_size, max_iter=max_iter, lb=[lb]*dim, ub=[ub]*dim)
    _, best_y = de.run()
    return float(np.squeeze(best_y))

def run_ga(func, dim, max_iter, pop_size, lb, ub):
    X = np.random.uniform(lb, ub, (pop_size, dim))
    Y = np.array([func(ind) for ind in X])
    best = Y.min()
    for _ in range(max_iter):
        parents = X[np.argsort(Y)[:pop_size//2]]
        children = []
        for i in range(0, len(parents), 2):
            if i+1 < len(parents):
                alpha = np.random.rand()
                c1 = alpha*parents[i] + (1-alpha)*parents[i+1]
                c2 = alpha*parents[i+1] + (1-alpha)*parents[i]
                children += [c1, c2]
        children = np.array(children)
        children += np.random.normal(0, 0.1, size=children.shape)
        children = np.clip(children, lb, ub)
        X = np.vstack([parents, children])[:pop_size]
        Y = np.array([func(ind) for ind in X])
        if Y.min() < best: best = Y.min()
    return float(best)

# === Results Directory ===
os.makedirs("Results", exist_ok=True)

# === Main Benchmark Runner ===
def main():
    max_iter, pop_size, trials = 300, 30, 20
    dimensions = [30, 50, 100]

    podium_scores = {"EGO": 0, "PSO": 0, "DE": 0, "GA": 0}

    for name, func in benchmarks.items():
        for dim in dimensions:
            print(f"\n=== Running on {name} ({dim}D) ===")
            lb, ub = -5.12, 5.12
            if name in ["Step", "Elliptic", "NoisyQuartic", "Levy", "FletcherPowell"]:
                lb, ub = -100, 100
            if name == "SchafferF6":
                dim = 2
                lb, ub = -100, 100

            algo_results = {"EGO": [], "PSO": [], "DE": [], "GA": []}
            runtimes = {"EGO": [], "PSO": [], "DE": [], "GA": []}
            conv_curves = {"EGO": []}

            for _ in range(trials):
                # EGO
                start = time.time()
                _, best_y, hist = entropy_guided_optimization(func, dim=dim, max_iter=max_iter, pop_size=pop_size)
                runtimes["EGO"].append(time.time() - start)
                algo_results["EGO"].append(best_y)
                conv_curves["EGO"].append(hist)

                # PSO
                start = time.time()
                algo_results["PSO"].append(run_pso(func, dim, max_iter, pop_size, lb, ub))
                runtimes["PSO"].append(time.time() - start)

                # DE
                start = time.time()
                algo_results["DE"].append(run_de(func, dim, max_iter, pop_size, lb, ub))
                runtimes["DE"].append(time.time() - start)

                # GA
                start = time.time()
                algo_results["GA"].append(run_ga(func, dim, max_iter, pop_size, lb, ub))
                runtimes["GA"].append(time.time() - start)

            # Save CSV
            df = pd.DataFrame({
                "EGO": algo_results["EGO"],
                "PSO": algo_results["PSO"],
                "DE": algo_results["DE"],
                "GA": algo_results["GA"],
                "EGO_time": runtimes["EGO"],
                "PSO_time": runtimes["PSO"],
                "DE_time": runtimes["DE"],
                "GA_time": runtimes["GA"],
            })
            df.to_csv(f"Results/{name}_{dim}D.csv", index=False)

            # Convergence plot
            plt.figure(figsize=(6, 5))
            mean_curve = np.mean(conv_curves["EGO"], axis=0)
            plt.plot(mean_curve, label="EGO")
            plt.title(f"{name} Convergence ({dim}D)")
            plt.xlabel("Iterations")
            plt.ylabel("Fitness")
            if np.all(mean_curve > 0): plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"Results/{name}_{dim}D_convergence.png")
            plt.close()

            # Bar chart
            means = {k: np.mean(v) for k, v in algo_results.items()}
            plt.figure(figsize=(6, 5))
            plt.bar(means.keys(), means.values())
            plt.title(f"{name} Final Fitness ({dim}D)")
            plt.ylabel("Fitness")
            plt.tight_layout()
            plt.savefig(f"Results/{name}_{dim}D_bars.png")
            plt.close()

            # Update podium
            winner = min(means, key=means.get)
            podium_scores[winner] += 1

    # Print podium
    print("\n=== Podium Scores (all funcs + dims) ===")
    for algo, score in podium_scores.items():
        print(f"{algo}: {score} wins")

if __name__ == "__main__":
    main()
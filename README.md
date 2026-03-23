# SEDE: Spatial Entropy Differential Evolution

**SEDE** is a production-grade, hyper-dimensional optimization metaheuristic. Built on the structural foundations of Differential Evolution (DE), SEDE integrates a bespoke **Spatial Entropy** measurement derived from an exact $O(N^2)$ Gaussian kernel to dynamically govern the boundaries between structural exploration ($CR$) and targeted exploitation ($F$) phases.

This repository natively features the newly compiled **C++ OpenMP Backend** (via `PyBind11`), achieving a definitive **~4.7x acceleration** over standard JIT-bound Python implementations. This scaling factor allows operations to rapidly traverse complex Machine Learning tuning matrices (like SVM continuous threshold sweeping) while entirely mitigating the notorious Boundary Stagnation logic up to $500$ dimensions.

---

## 🚀 Key Mathematical Reforms (v2)
To surpass standard boundaries at ultra-high dimensionality ($D > 100$), the algorithm internally employs 4 critical breakthroughs:

1. **Genotypic (Spatial) Entropy**: Replaced legacy fitness-based metrics with exact euclidean structural distances, protecting the core engine from "Phenotypic Entropy Paradoxes" (where algorithmic fitness plateaus falsely signal convergence).
2. **Dynamic Logistic Governance ($\Delta H$)**: Real-time entropy mapping triggers exponential decay curves for the Mutation Generation. Rapid negative entropy drops trigger massive forced recombination grids to escape local wells.
3. **Power-Law Damping**: At extreme dimensionality limits, maximum vector jumps ($F_{max}$) strictly decay using a uniform power-law configuration restricting vector overflow.
4. **Coordinate-Wise Clamping (Evolutionary Cooling)**: Evaluates vector vectors iteratively against the "Empty Space Phenomenon". The displacement size is strictly bounded to an absolute limit that scales dynamically via linear decay from $5\% \rightarrow 0.1\%$ of the true bounds over generation runtime.

---

## 🛠️ Direct Installation (Local)

To embed the matrix natively into your local environment:
1. Ensure you have Python $3.11+$ installed.
2. Install `pybind11` and run the integrated wheel. 

```bash
# Build the native PyBind11 C++ engine across all cores
pip install pybind11
pip install .
```
*(Note for macOS Users: The internal `setup.py` wheel actively links `-lomp` pathways resolving missing OpenMP configurations safely!).*

---

## 🐳 Dynamic Docker Deployment (Recommended)
For an isolated and totally resilient environment executing the compiled logic identically, build the integrated `Dockerfile`! This handles all environment prerequisites automatically.

The Docker container runs our standalone `runner.py`, explicitly designed to accept parameter tuning out-of-the-box!

```bash
# 1. Build the Accelerated C++ Container Image
docker build -t sede_cli .

# 2. Execute natively across the Dockerized CLI
docker run --rm sede_cli --func sphere --dim 500 --iter 1000 --pop 50
```

### CLI Arguments 
- `--func`: String identifier (`sphere`, `rosenbrock`, `rastrigin`)
- `--dim`: Swarm Dimensionality (e.g., `100`, `500`)
- `--iter`: Absolute Max Generations (e.g., `1000`)
- `--pop`: Active Population (e.g., `50`)

---

## 📊 Analytics & Benchmarking (Developer)
If you wish to recalculate the definitive 30-Run Baselines validating SEDE mathematically against standard Particle Swarm operations (PSO) and legacy DE:

- **Speed-Gap Verification**:
  Proves the explicit Python Numba $O(N^2)$ compilation vs the current C++ $O(N^2)$ Engine.
  ```bash
  python benchmarks/Speed_Gap_Test.py
  ```

- **Real-World HPO Testing (Support Vector Machines)**:
  Executes a functional 5-Fold validation test across Breast Cancer hyperparameter tuning boundaries plotting performance vs iterations.
  ```bash
  python benchmarks/RealWorldBench.py
  ```

- **Deep Matrix Convergence Suite**:
  Aggregates robust massive baseline plots logging $10D, 50D, 100D$ evaluations across 30 identical permutations outputting High-Res Graph arrays (`Results/*.png`).
  ```bash
  python benchmarks/Final_Transition.py
  ```

*Created as part of the hyper-dimensional optimization tuning transition protocol.*

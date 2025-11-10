# Entropy-Guided Optimization (EGO)

© 2025 **Azer Adham** — All rights reserved.

Developed and maintained by Azer Adham (Düzce University, Computer Engineering).

---

## Overview
Entropy-Guided Optimization (EGO) is a novel evolutionary algorithm that adaptively balances exploration and exploitation through an entropy-weighted mutation and crossover scheme. It draws on principles of information theory and Differential Evolution to dynamically adjust search diversity.

This repository contains the original implementation, benchmark framework, and statistical analysis used to evaluate EGO against PSO, DE, and GA across 14+ classical benchmark functions and multiple dimensions (30D, 50D, 100D).

---

## Repository Structure
```
EGO.py              → Core algorithm implementation
Test.py             → Benchmark runner and performance comparison
Analyze_data.py     → Statistical tests (Friedman / Wilcoxon) + rank tables + CD diagram
Pseudocode.txt      → Formal algorithm pseudocode
Results/            → CSV data and convergence / bar-chart PNGs
paper/              → (optional) LaTeX draft of the academic paper
```

---

## Key Features
- Adaptive entropy weighting mechanism
- Hybrid DE-style mutation + crossover operators
- Comprehensive benchmark coverage (Sphere, Rosenbrock, Rastrigin, Ackley, etc.)
- Automatic result visualization and LaTeX table export
- Statistical validation via Friedman and Wilcoxon tests

---

## Requirements
```bash
Python >= 3.10
pip install numpy pandas matplotlib scikit-opt scipy
```

---

## Running Benchmarks
```bash
python Test.py
```
Outputs are saved under `Results/`:
- `*_convergence.png` (mean convergence curves)
- `*_bars.png` (final fitness mean ± std)
- `<func>_<dim>.csv` (raw trial results)

---

## Statistical Analysis
After running the benchmarks:
```bash
python Analyze_data.py
```
Generates:
- Friedman & Wilcoxon test outputs (console)
- `ranks_table.tex` and `results_table.tex`
- `cd_diagram.png` (Critical Difference diagram)

---

## Reproducibility
- Fixed random seeds per trial in `Test.py`
- All bounds and dimensions are explicitly logged
- CSVs include algorithm-wise final fitness per trial

---

## Citation
If referencing this work:

> Adham, A. (2025). *Entropy-Guided Optimization (EGO): A Hybrid Entropy-Driven Metaheuristic Algorithm.* Düzce University, Computer Engineering Department.

---

## License
See `LICENSE`. Unauthorized distribution or derivative publication is prohibited without written permission.

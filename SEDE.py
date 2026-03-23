import numpy as np
from numba import njit

# ==========================================
# 1. JIT Compiled Spatial Entropy (C-Speeds)
# ==========================================
@njit(fastmath=True)
def compute_spatial_entropy_numba(X, N):
    """
    Calculates Genotypic (Spatial) Shannon entropy using explicit 
    for-loops. Numba compiles this to raw machine code, bypassing 
    Python's interpreter tax for massive speedups.
    """
    d_matrix = np.zeros((N, N))
    sum_d = 0.0
    count = 0.0
    
    # Calculate exact Euclidean distance matrix
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = np.sqrt(np.sum((X[i] - X[j])**2))
                d_matrix[i, j] = dist
                sum_d += dist
                count += 1.0
                
    # Mean distance for dynamic Gaussian bandwidth
    mean_d = sum_d / count if count > 0 else 1e-8
    sigma = mean_d + 1e-15
    
    # Local Spatial Density
    rho = np.zeros(N)
    for i in range(N):
        for j in range(N):
            rho[i] += np.exp(-(d_matrix[i, j]**2) / (2.0 * sigma**2))
            
    # Normalize to probability
    rho_sum = np.sum(rho)
    P = np.zeros(N)
    
    if rho_sum > 1e-15:
        for i in range(N):
            P[i] = rho[i] / rho_sum
            # Clip for log safety
            if P[i] < 1e-15:
                P[i] = 1e-15
    else:
        # Fallback to uniform if perfectly converged
        for i in range(N):
            P[i] = 1.0 / N
            
    # Spatial Shannon Entropy
    H_spatial = 0.0
    for i in range(N):
        H_spatial -= P[i] * np.log(P[i])
        
    return H_spatial

# ==========================================
# 2. Main SEDE Class
# ==========================================
class SEDE:
    """
    Spatial Entropy Differential Evolution (SEDE).
    
    Mechanism:
    Uses Spatial Entropy based on population geometry to adaptively control 
    the Mutation Factor (F) and Crossover Rate (CR) using a non-linear logistic curve.
    - High Spatial Entropy -> High F/CR (Exploration)
    - Low Spatial Entropy  -> Low F/CR (Exploitation)
    """
    def __init__(self, func, dim, pop_size=50, max_iter=1500, lb=-100, ub=100,
                 seed=None, k=3.0, F_min=0.2, F_max=0.9, CR_base=0.9, CR_spike=0.9, **kwargs):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.array([lb] * dim)
        self.ub = np.array([ub] * dim)
        self.rng = np.random.default_rng(seed)

        # Hyperparameters for Non-Linear Mapping
        self.k = k
        self.F_max = 0.9 * (self.dim ** -0.15)
        self.F_min = 0.1 * (self.dim ** -0.15)
        # H_mid is the inflection point (midpoint of possible entropy) max_entropy is np.log(N)
        self.H_mid = np.log(self.pop_size) / 3.0 
        
        # Hyperparameters for CR Strategy
        self.CR_base = CR_base
        self.CR_spike = CR_spike
        self.entropy_drop_threshold = -0.05 * (2.0 * self.H_mid) # 5% drop of max entropy threshold

        # Initialize Population
        self.X = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.Y = np.array([self.func(x) for x in self.X])

        # Track Best
        self.best_idx = np.argmin(self.Y)
        self.best_x = self.X[self.best_idx].copy()
        self.best_y = self.Y[self.best_idx]
        self.history = [self.best_y]

    def optimize(self):
        prev_H_spatial = None
        
        for t in range(self.max_iter):
            # --- 1. Calculate Spatial Entropy State (Using Numba) ---
            H_spatial = compute_spatial_entropy_numba(self.X, self.pop_size)
            
            if prev_H_spatial is None:
                delta_H = 0.0
            else:
                delta_H = H_spatial - prev_H_spatial
                
            prev_H_spatial = H_spatial
            
            # --- 2. Adaptive Parameters (Non-Linear Logistic Mapping) ---
            # Pure logistic curve without the double-exponential contraction penalty
            current_H_mid = self.H_mid * (1.0 - (t / self.max_iter))
            H_norm = H_spatial / np.log10(self.dim + 9)
            logistic_term = 1.0 / (1.0 + np.exp(-self.k * (H_norm - current_H_mid)))
            current_F = self.F_min + (self.F_max - self.F_min) * logistic_term
            
            # Separate Feedback loop for CR_new (Entropy Differential Jump)
            if delta_H < self.entropy_drop_threshold:
                current_CR = self.CR_spike  # Massive genetic recombination due to rapid convergence
            else:
                current_CR = self.CR_base   # Stable baseline
            
            # --- 3. Mutation (DE/rand-to-best/1) ---
            # Continuous blending of exploration and exploitation
            idxs = self.rng.integers(0, self.pop_size, size=(self.pop_size, 3))
            a = self.X[idxs[:, 0]]
            b = self.X[idxs[:, 1]]
            c = self.X[idxs[:, 2]]
            
            # V = random_base + pull_to_best + difference_vector
            V_mutant = a + current_F * (self.best_x - a) + current_F * (b - c)
            
            # --- Mutation Step-Clipping ---
            # Evolutionary Cooling: Linear Decay from 5% to 0.1%
            progress = t / self.max_iter
            current_percent = 0.05 - (0.05 - 0.001) * progress
            delta = current_percent * (self.ub - self.lb)
            
            # Coordinate-Wise Clamping
            displacement = np.clip(V_mutant - a, -delta, delta)
            V_mutant = a + displacement
            
            # --- 4. Crossover ---
            cross_mask = self.rng.random(size=(self.pop_size, self.dim)) < current_CR
            U = np.where(cross_mask, V_mutant, self.X)
            
            # --- 5. Bounds & Selection ---
            U = np.clip(U, self.lb, self.ub)
            Y_new = np.array([self.func(u) for u in U])
            
            # Greedy Selection
            improved_mask = Y_new < self.Y
            self.X[improved_mask] = U[improved_mask]
            self.Y[improved_mask] = Y_new[improved_mask]
            
            # Update Global Best
            current_min = np.min(self.Y)
            if current_min < self.best_y:
                self.best_y = current_min
                self.best_x = self.X[np.argmin(self.Y)].copy()
                
            self.history.append(self.best_y)
            
        return self.best_x, self.best_y, self.history

# ==========================================
# 3. Wrapper Function
# ==========================================
def spatial_entropy_differential_evolution(func, dim, pop_size=50, max_iter=1500, lb=-100, ub=100, 
                                           F=None, mutation_rate=None, seed=None):
    optimizer = SEDE(func, dim, pop_size, max_iter, lb, ub, seed=seed)
    return optimizer.optimize()
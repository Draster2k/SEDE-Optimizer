import numpy as np

class EGO:
    def __init__(self, func, dim, pop_size=30, max_iter=300, lb=-5.12, ub=5.12,
                 F=0.5, mutation_rate=0.2, seed=None):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb)
        self.ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub)
        self.F = F
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)

        # Initialize population
        self.X = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.Y = np.array([self.func(x) for x in self.X])

        # Track best
        self.best_idx = np.argmin(self.Y)
        self.best_x = self.X[self.best_idx].copy()
        self.best_y = self.Y[self.best_idx]

        self.history = [self.best_y]

    def entropy(self, probs):
        """Shannon entropy of probability distribution."""
        probs = np.clip(probs, 1e-12, 1)  # avoid log(0)
        return -np.sum(probs * np.log(probs))

    def selection_probs(self, fitness):
        """Convert fitness to selection probabilities (lower is better)."""
        fitness = fitness - fitness.min() + 1e-12
        probs = 1.0 / fitness
        probs /= probs.sum()
        return probs

    def mutate_de(self, idx):
        """DE/rand/1 mutation operator."""
        ids = list(range(self.pop_size))
        ids.remove(idx)
        a, b, c = self.rng.choice(ids, 3, replace=False)
        mutant = self.X[a] + self.F * (self.X[b] - self.X[c])
        return np.clip(mutant, self.lb, self.ub)

    def run(self):
        for t in range(self.max_iter):
            # Adaptive entropy weight (high early, low later)
            entropy_weight = max(0.1, 1.0 - t / self.max_iter)

            # Selection probabilities
            probs = self.selection_probs(self.Y)
            entropy_val = self.entropy(probs)

            new_X = []
            for i in range(self.pop_size):
                if self.rng.random() < self.mutation_rate:
                    # DE-style mutation
                    child = self.mutate_de(i)
                else:
                    # Crossover between two parents
                    p1, p2 = self.rng.choice(self.pop_size, 2, replace=False)
                    alpha = self.rng.random()
                    child = alpha * self.X[p1] + (1 - alpha) * self.X[p2]

                # Inject exploration: perturb with entropy weight
                noise = self.rng.normal(0, entropy_weight, size=self.dim)
                child = child + noise

                # Clip to bounds
                child = np.clip(child, self.lb, self.ub)
                new_X.append(child)

            # Evaluate
            new_X = np.array(new_X)
            new_Y = np.array([self.func(x) for x in new_X])

            # Elitism: keep best
            if new_Y.min() < self.best_y:
                self.best_idx = new_Y.argmin()
                self.best_x = new_X[self.best_idx].copy()
                self.best_y = new_Y[self.best_idx]

            # Replace if better, otherwise keep old
            improved = new_Y < self.Y
            self.X[improved] = new_X[improved]
            self.Y[improved] = new_Y[improved]

            self.history.append(self.best_y)

        return self.best_x, self.best_y, self.history


def entropy_guided_optimization(func, dim, pop_size=30, max_iter=300, lb=-5.12, ub=5.12,
                                F=0.5, mutation_rate=0.2, seed=None):
    ego = EGO(func, dim=dim, pop_size=pop_size, max_iter=max_iter,
              lb=lb, ub=ub, F=F, mutation_rate=mutation_rate, seed=seed)
    best_x, best_y, history = ego.run()
    return best_x, best_y, history
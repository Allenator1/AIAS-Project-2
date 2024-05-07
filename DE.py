import numpy as np


class Target:
    """
    Represents an individual within the population in DE
    """
    def __init__(self, bounds, vector=None):
        self.bounds = bounds
        if vector is not None:
            self.vector = vector
        else:
            self.vector = np.random.uniform(bounds[:, 0], bounds[:, 1])


class DE:
    """
    Original Differential Evolution algorithm.
    """
    def __init__(self, objective_function, bounds, 
              initial_population, max_iter=1000, F=0.5, CR=0.9):
        self.objective_function = objective_function
        self.bounds = bounds
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        self.generation = 0
        self.population = initial_population
        self.NP = len(initial_population)


    def iterate(self):
        """Perform a single iteration of DE."""
        for i in range(self.NP):
            # Select three random indices
            r1, r2, r3 = np.random.choice(self.NP, 3, replace=False)
            target = self.population[i]
            target_vector = target.vector

            # Mutation
            mutant_vector = self.population[r1].vector + self.F * (self.population[r2].vector - self.population[r3].vector)

            # Crossover
            cross_points = np.random.rand(len(mutant_vector)) < self.CR
            trial_vector = np.where(cross_points, mutant_vector, target_vector)

            # Clip to bounds
            trial_vector = np.clip(trial_vector, self.bounds[:, 0], self.bounds[:, 1])
            trial = self.population[i].__class__(self.bounds, trial_vector)

            # Selection
            if self.objective_function(trial) < self.objective_function(target):
                self.population[i] = trial
        self.generation += 1

    
    def get_best(self):
        """Return the index of the best individual in the population."""
        return self.population[np.argmin([self.objective_function(x) for x in self.population])]
    

# Test usage on the Griewangk function
if __name__ == '__main__':
    def griewangk_function(x):
        vec = x.vector
        j = np.arange(1, len(vec) + 1)
        return np.sum(vec**2 / 4000) - np.prod(np.cos(vec / np.sqrt(j))) + 1

    bounds = np.array([[-400, 400] for _ in range(10)])
    initial_pop = [Target(bounds) for _ in range(25)]

    de = DE(griewangk_function, bounds, initial_pop, max_iter=20000, F=0.5, CR=0.25)

    average_fitness = np.inf
    while average_fitness > 1e-5 and de.generation < de.max_iter:
        de.iterate()
        average_fitness = np.mean([griewangk_function(x) for x in de.population])
        print(f'Generation {de.generation}: {average_fitness}')

    best_target = de.get_best()
    print(f'Best vector: {best_target.vector}')
    print(f'Best fitness: {griewangk_function(best_target)}')

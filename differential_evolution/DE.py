import numpy as np


class DE:
	"""
    Original Differential Evolution algorithm.
    """
	def __init__(self, objective_function, bounds, NP=10, max_iter=1000, F=0.5, CR=0.9):
		self.objective_function = objective_function
		self.bounds = bounds
		self.NP = NP
		self.max_iter = max_iter
		self.F = F
		self.CR = CR
		self.population = [np.random.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(NP)]
		self.generation = 0


	def iterate(self):
		"""Perform a single iteration of DE."""
		for i in range(self.NP):
			# Select three random indices
			r1, r2, r3 = np.random.choice(self.NP, 3, replace=False)
			target = self.population[i]

			# Mutation
			mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])

			# Crossover
			cross_points = np.random.rand(len(mutant)) < self.CR
			trial = np.where(cross_points, mutant, target)

			# Clip to bounds
			trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])

			# Selection
			if self.objective_function(trial) < self.objective_function(target):
				self.population[i] = trial
		self.generation += 1

	
	def get_best(self):
		"""Return the index of the best individual in the population."""
		return np.argmin([self.objective_function(x) for x in self.population])
	

# Test usage
if __name__ == '__main__':
	def griewangk_function(x):
		j = np.arange(1, len(x) + 1)
		return np.sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(j))) + 1

	bounds = np.array([[-400, 400] for _ in range(10)])

	de = DE(griewangk_function, bounds, NP=25, max_iter=20000, F=0.5, CR=0.2)

	average_fitness = np.inf
	while average_fitness > 1e-5 and de.generation < de.max_iter:
		de.iterate()
		average_fitness = np.mean([griewangk_function(x) for x in de.population])
		print(f'Generation {de.generation}: {average_fitness}')

	best_vector = de.population[de.get_best()]
	print(f'Best vector: {best_vector}')
	print(f'Best fitness: {griewangk_function(best_vector)}')



class GA:
    """
    Genetic Algoritm implementaion
    """

    def __init__(self, objective_function, bounds, initial_population,
                 clew_function, max_iter = 100):
        self.objective_function = objective_function
        self.clew_function = clew_function
        self.bounds = bounds
        self.max_iter = max_iter
        self.generation = 0
        self.populations = initial_population
        self.NP = len(initial_population)
        self.NW = len(initial_population[0])

    def iterate(self):
        """Perform single iteration of GA"""
    for i in range(self.NP):
        
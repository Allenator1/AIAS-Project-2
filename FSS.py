import numpy as np 

INITIAL_WT = 10


class Fish:
    """
    Represents an individual within the population in FSS
    """
    def __init__(self, target):
        self.target = target
        self.wt = INITIAL_WT                        # Weight of the fish
        self.df = 0                                 # Change in fitness of the fish from swimming              
        self.delta_pos = np.zeros(target.shape)     # Change in position of the fish from swimming

    @property
    def shape(self):
        return self.vector.shape
    
    @property
    def vector(self):
        return self.target.vector
    
    @vector.setter
    def vector(self, value):
        self.target.vector = value


class FSS:
    """
    Fish School Search algorithm.
    """
    def __init__(self, fit_func, bounds, initial_population, 
                 school_weight=0, step_ind=0.01, step_vol=0.01, max_iter=1000):
        self.fit_func = fit_func
        self.bounds = bounds
        self.max_iter = max_iter
        self.generation = 0
        self.population = [Fish(x) for x in initial_population]
        self.step_ind = bounds[:, 1] - bounds[:, 0] * step_ind
        self.step_vol = (bounds[:, 1] - bounds[:, 0]) * step_vol
        self.previous_school_wt = school_weight


    def iterate(self):
        """Perform a single iteration of FSS."""
        # Update the position of each fish randomly using the swim function
        max_df = 0
        for f in self.population:
            self.swim_fish(f)
            if f.df > max_df:
                max_df = f.df
        
        # Apply the feeding operator
        school_wt = 0
        for f in self.population:
            self.feed_fish(f, max_df)
            school_wt += f.wt
        self.school_wt_improved = school_wt > self.previous_school_wt

        # Calculate intinctive movement as the weighted average of fish movement
        instinctive_move = np.zeros(self.population[0].shape)
        sum_df = 0
        for f in self.population:
            instinctive_move += f.delta_pos * f.df
            sum_df += f.df
        instinctive_move /= sum_df

        # Calculate the barycenter coordinates at the weighted average of fish positions
        barycenter = np.zeros(self.population[0].shape)
        sum_wt = 0
        for f in self.population:
            barycenter += f.vector * f.wt
            sum_wt += f.wt
        barycenter /= sum_wt

        # Apply instinctive and volitive movement
        for f in self.population:
            self.instinctive_movement(f, instinctive_move)
            self.volitive_movement(f, barycenter)

        # Update the generation
        for i in range(self.NP):
            f = self.population[i]
            f.target = f.target.__class__(self.bounds, f.vector)


    def swim_fish(self, f):
        new_vec = f.vector + np.random.uniform(-1, 1, f.shape) * self.step_ind
        new_vec = np.clip(new_vec, self.bounds[:, 0], self.bounds[:, 1])

        new_target = f.target.__class__(self.bounds, new_vec)
        df = self.fit_func(new_target) - self.fit_func(f.target)

        if df >= 0:
            f.df = df
            f.delta_pos = new_vec - f.vector
            f.target = new_target
    

    def feed_fish(self, f, max_df):
        f.wt = max(INITIAL_WT, f.wt + f.df / max_df)

    
    def instinctive_movement(self, f, move_vector):
        f.vector += move_vector
        f.vector = np.clip(f.vector, self.bounds[:, 0], self.bounds[:, 1])

    
    def volitive_movement(self, f, barycenter):
        dist2bary = np.linalg.norm(barycenter - f.vector)
        move_vector = self.step_vol * np.random.uniform(0, 1, f.shape) * (f.vector - barycenter) * dist2bary
        if self.school_wt_improved:
            move_vector = -move_vector    # Move towards the barycenter for exploitation  
        f.vector += move_vector
        f.vector = np.clip(f.vector, self.bounds[:, 0], self.bounds[:, 1])
        

    def get_best(self):
        """Return the index of the best individual in the population."""
        return self.population[np.argmin([self.fit_func(x) for x in self.population])]
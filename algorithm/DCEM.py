'''
The Differentiable Cross-Entropy Method - ICML 2020
'''
import numpy as np
from pypop7.optimizers.cem.cem import CEM
from pypop7.optimizers.cem.dcem import DCEM

class Shallow_DCEM(DCEM):
    def optimize(self, fitness_function=None, args=None):
        fitness = CEM.optimize(self, fitness_function)
        mean, x, y = self.initialize()
        while True:
            x, y = self.iterate(mean, x, y, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            self._print_verbose_info(y)
            self._n_generations += 1
            mean = self.update_distribution(x, y)
        return self._collect_results(fitness, mean)
'''
MMES: Mixture Model-Based Evolution Strategy for Large-Scale Optimization - TEVC 2021
'''
import numpy as np
from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.mmes import MMES

class Shallow_MMES(MMES):
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, w, q, t, v, y = self.initialize(args)
        if self.saving_fitness:
            fitness.append(y[0])
        self._print_verbose_info(y)
        while True:
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            x, y = self.iterate(x, mean, q, v, y, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p, w, q, t, v = self._update_distribution(x, mean, p, w, q, t, v, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(y)
            if self.is_restart:
                x, mean, p, w, q, t, v, y = self.restart_reinitialize(
                    args, x, mean, p, w, q, t, v, y, fitness)
        results = self._collect_results(fitness, mean)
        results['p'] = p
        results['w'] = w
        return results
'''
A computationally efficient limited memory CMA-ES for large scale optimization - GECCO 2014
'''
import numpy as np
from pypop7.optimizers.es.es import ES
from pypop7.optimizers.es.lmcmaes import LMCMAES

class Shallow_LMCMAES(LMCMAES):
    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        mean, x, p_c, s, vm, pm, b, d, y = self.initialize(args)
        while True:
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            x, y = self.iterate(mean, x, pm, vm, y, b, args)
            if self.saving_fitness:
                fitness.extend(y)
            if self._check_terminations():
                break
            mean, p_c, s, vm, pm, b, d = self._update_distribution(
                mean, x, p_c, s, vm, pm, b, d, y, y_bak)
            self._print_verbose_info(y)
            self._n_generations += 1
            if self.is_restart:
                mean, x, p_c, s, vm, pm, b, d, y = self.restart_reinitialize(
                    args, mean, x, p_c, s, vm, pm, b, d, y)
        results = self._collect_results(fitness, mean)
        results['p_c'] = p_c
        results['s'] = s
        return results
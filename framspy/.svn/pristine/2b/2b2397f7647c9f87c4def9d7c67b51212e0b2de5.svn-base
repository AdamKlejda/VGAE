from typing import Dict, List

from evolalg.base.individual import Individual
from evolalg.fitness.fitness_step import FitnessStep
import copy

class MultipleEvaluationFitness(FitnessStep):
    def __init__(self, frams_lib, fields: Dict, fields_defaults: Dict, commands: List[str] = None,
                 vectorized: bool = True, number_evaluations = 10, *args, **kwargs):
        super().__init__(frams_lib, fields, fields_defaults, commands, vectorized, *args, **kwargs)
        self.number_evaluations = number_evaluations


    def call(self, population: List[Individual]):
        super(MultipleEvaluationFitness, self).call(population)
        pop = copy.deepcopy(population)
        statistics = [{} for _ in range(len(population))]
        for _ in self.number_evaluations:
            pass
        return population

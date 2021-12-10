import random

from evolalg.base.frams_step import FramsStep
from evolalg.base.individual import Individual


class FramsMutation(FramsStep):
    def __init__(self, frams_lib, commands, mutate_prob, *args, **kwargs):
        super().__init__(frams_lib, commands, *args, **kwargs)
        self.mutate_prob = mutate_prob

    def call(self, population):
        super(FramsMutation, self).call(population)
        idx = []
        for i in range(len(population)):
            if random.random() < self.mutate_prob:
                idx.append(i)
        mutated = [population[_].genotype for _ in idx]
        mutated = self.frams.mutate([_ for _ in mutated])

        for i, m in zip(idx, mutated):
            population[i] = Individual(m)

        return population

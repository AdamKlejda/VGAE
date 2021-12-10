from evolalg.base.frams_step import FramsStep
from evolalg.mutation_cross.frams_cross import FramsCross
from evolalg.mutation_cross.frams_mutation import FramsMutation
import random


class FramsCrossOrMutate(FramsStep):
    def __init__(self, frams_lib, commands, cross_prob, mutate_prob, mutate_commands=None, cross_commands=None, *args,
                 **kwargs):
        super().__init__(frams_lib, commands, *args, **kwargs)

        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob

        self.frams_mutate = FramsMutation(frams_lib, mutate_commands, 1)
        self.frams_cross = FramsCross(frams_lib, cross_commands, 1)

    def call(self, population):
        super(FramsCrossOrMutate, self).call(population)
        mutate_idx = []
        cross_idx = []
        for i in range(len(population)):
            rng = random.random()
            if rng < self.cross_prob:
                cross_idx.append(i)
            elif rng < self.cross_prob + self.mutate_prob:
                mutate_idx.append(i)

        mutate_ind = [population[_] for _ in mutate_idx]
        cross_ind = [population[_] for _ in cross_idx]

        if len(mutate_ind) > 0:
            self.frams_mutate(mutate_ind)
        if len(cross_idx) > 0:
            self.frams_cross(cross_idx)

        for i, ind in zip(mutate_idx, mutate_ind):
            population[i] = ind

        for i, ind in zip(cross_idx, cross_ind):
            population[i] = ind

        return population

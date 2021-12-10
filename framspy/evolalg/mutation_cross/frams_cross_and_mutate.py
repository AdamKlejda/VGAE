from evolalg.base.frams_step import FramsStep
from evolalg.mutation_cross.frams_cross import FramsCross
from evolalg.mutation_cross.frams_mutation import FramsMutation


class FramsCrossAndMutate(FramsStep):
    def __init__(self, frams_lib, cross_prob, mutate_prob, mutate_commands=None, cross_commands=None, *args, **kwargs):
        super().__init__(frams_lib, *args, **kwargs)

        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob

        self.frams_mutate = FramsMutation(frams_lib, mutate_commands, mutate_prob)
        self.frams_cross = FramsCross(frams_lib, cross_commands, cross_prob)

    def call(self, population):
        super(FramsCrossAndMutate, self).call(population)
        population = self.frams_cross(population)
        population = self.frams_mutate(population)
        return population

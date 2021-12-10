import random

from evolalg.base.frams_step import FramsStep
from evolalg.base.individual import Individual


class FramsCross(FramsStep):
    def __init__(self, frams_lib, commands, cross_prob, *args, **kwargs):
        super().__init__(frams_lib, commands, *args, **kwargs)
        self.cross_prob = cross_prob

    def call(self, population):
        super(FramsCross, self).call(population)
        for i in range(1, len(population), 2):
            if random.random() < self.cross_prob:
                geno1 = population[i - 1].genotype
                geno2 = population[i].genotype

                cross_geno1 = self.frams.crossOver(geno1, geno2)
                cross_geno2 = self.frams.crossOver(geno1, geno2)

                population[i - 1] = Individual(cross_geno1)
                population[i] = Individual(cross_geno2)

        return population

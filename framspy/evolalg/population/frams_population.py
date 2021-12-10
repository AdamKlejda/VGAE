from evolalg.base.frams_step import FramsStep
from evolalg.base.individual import Individual


class FramsPopulation(FramsStep):
    def __init__(self, frams_lib, genetic_format, pop_size, commands=None, *args, **kwargs):

        if commands is None:
            commands = []
        super().__init__(frams_lib, commands,*args, **kwargs)

        self.pop_size = pop_size
        self.genetic_format = genetic_format

    def call(self, population, *args, **kwargs):
        super(FramsPopulation, self).call(population)
        return [Individual(self.frams.getSimplest(self.genetic_format)) for _ in range(self.pop_size)]


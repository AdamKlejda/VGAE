from abc import abstractmethod
from collections import Iterable

from evolalg.base.step import Step


class Repair(Step):
    def __init__(self, excepted_size, *args, **kwargs):
        super(Repair, self).__init__(*args, **kwargs)
        self.excepted_size = excepted_size

    @abstractmethod
    def generate_new(self, population, missing_count):
        pass

    def call(self, population):
        super(Repair, self).call(population)
        generated = []
        while len(generated) + len(population) < self.excepted_size:
            gen = self.generate_new(population, self.excepted_size-len(population)-len(generated))
            if isinstance(gen, Iterable):
                generated.extend(gen)
            else:
                generated.append(gen)
        population.extend(generated)
        return population[:self.excepted_size]

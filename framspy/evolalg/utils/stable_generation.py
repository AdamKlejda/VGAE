from collections import Iterable

from evolalg.base.step import Step
import copy

from evolalg.base.union_step import UnionStep


class StableGeneration(Step):

    def __init__(self, selection, steps, population_size=None):
        self.selection = selection
        self.steps = UnionStep(steps)
        self.population_size = population_size

    def generate_new(self, population, missing_count):
        selected = self.selection(population, missing_count)
        selected = copy.deepcopy(selected)
        selected = self.steps(selected)
        return selected

    def init(self):
        self.selection.init()
        self.steps.init()

    def call(self, population):
        population_size = self.population_size
        if population_size is None:
            population_size = len(population)
        generated = []
        while len(generated) < population_size:
            gen = self.generate_new(population, population_size - len(generated))
            if isinstance(gen, Iterable):
                generated.extend(gen)
            else:
                generated.append(gen)
        return generated[:len(population)]

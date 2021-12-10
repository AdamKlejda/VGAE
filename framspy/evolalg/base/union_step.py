from collections import Iterable

from evolalg.base.step import Step


class UnionStep(Step):
    def __init__(self, steps, *args, **kwargs):
        super(UnionStep, self).__init__(*args, **kwargs)
        if isinstance(steps, Iterable):
            self.steps = steps
        else:
            self.steps = [steps]

    def init(self):
        for s in self.steps:
            if isinstance(s, Step):
                s.init()

    def call(self, population):
        super(UnionStep, self).call(population)
        for s in self.steps:
            population = s(population)
        return population

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)
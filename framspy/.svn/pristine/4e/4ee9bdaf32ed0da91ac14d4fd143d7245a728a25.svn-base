from typing import Dict, List

from evolalg.base.frams_step import FramsStep
from evolalg.base.individual import Individual
import frams


class FitnessStep(FramsStep):
    def __init__(self, frams_lib, fields: Dict, fields_defaults: Dict, commands: List[str] = None,
                 vectorized: bool = True, evaluation_count=None, *args, **kwargs):

        super().__init__(frams_lib, commands, *args, **kwargs)
        self.fields = fields
        self.fields_defaults = fields_defaults
        self.vectorized = vectorized
        self.evaluation_count = evaluation_count
        self.evaluation_count_original = None  # to be able to restore to original value after it is changed

    def pre(self):
        if self.evaluation_count is not None:
            self.evaluation_count_original = frams.ExpProperties.evalcount._value()  # store original value and restore it in post()
            frams.ExpProperties.evalcount = self.evaluation_count

    def post(self):
        if self.evaluation_count is not None:
            frams.ExpProperties.evalcount = self.evaluation_count_original
            self.evaluation_count_original = None

    def call(self, population: List[Individual]):
        super(FitnessStep, self).call(population)
        if self.vectorized:
            data = self.frams.evaluate([_.genotype for _ in population])
        else:
            data = [self.frams.evaluate([_.genotype]) for _ in population]

        for ind, d in zip(population, data):
            for k, v in self.fields.items():
                try:
                    setattr(ind, v, d["evaluations"][""][k])
                except:
                    setattr(ind, v, self.fields_defaults[k])
        return population

from abc import abstractmethod

from evolalg.base.step import Step
import copy


class Selection(Step):
    def __init__(self, copy=False, *args, **kwargs):
        super(Selection, self).__init__(*args, **kwargs)
        self.copy = copy

    @abstractmethod
    def select_next(self, population):
        pass

    def call(self, population, count=None):
        super(Selection, self).call(population)
        res = []
        if count is None:
            count = len(population)

        for _ in range(count):
            sel = self.select_next(population)
            if self.copy:
                sel = copy.deepcopy(sel)
            res.append(sel)
        return res

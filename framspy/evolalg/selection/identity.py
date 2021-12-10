import copy

from evolalg.base.step import Step
from evolalg.selection.selection import Selection


class IdentitySelection(Selection):
    def __init__(self, copy=False, *args, **kwargs):
        super(IdentitySelection, self).__init__(copy, *args, **kwargs)

    def call(self, population, selection_size=None):
        super(IdentitySelection, self).call(population)
        res = population
        if selection_size is not None:
            res = population[:selection_size]

        if self.copy:
            res = copy.deepcopy(res)
        return res

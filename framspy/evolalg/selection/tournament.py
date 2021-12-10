from evolalg.base.individual import Individual
from typing import List
import random

from evolalg.base.step import Step
from evolalg.selection.selection import Selection


class TournamentSelection(Selection):
    def __init__(self, tournament_size: int, fit_attr="fitness", copy=False, *args, **kwargs):
        super(TournamentSelection, self).__init__(copy, *args, **kwargs)
        self.tournament_size = tournament_size
        self.fit_attr = fit_attr

    def select_next(self, population):
        selected = [random.choice(population) for i in range(self.tournament_size)]
        return max(selected, key=lambda x: getattr(x, self.fit_attr))

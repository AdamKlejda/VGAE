from evolalg.selection.selection import Selection
from deap import tools
import copy


class NSGA2Selection(Selection):
    def __init__(self, copy=False, *args, **kwargs):
        super(NSGA2Selection, self).__init__(*args, **kwargs, copy=copy)

    def call(self, population, count=None):
        super(NSGA2Selection, self).call(population)
        pop = tools.selNSGA2(population, len(population))
        # make count divisible by 4, required by deap.tools.selTournamentDCD()
        remainder=count % 4
        if remainder > 0:
            count += 4-remainder   
        return copy.deepcopy(tools.selTournamentDCD(pop, count)) # "The individuals sequence length has to be a multiple of 4 only if k is equal to the length of individuals"

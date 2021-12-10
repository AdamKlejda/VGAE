import copy
import random

from evolalg.repair import Repair


class HallOfFameRepair(Repair):
    def __init__(self, excepted_size, halloffame, top, *args, **kwargs):
        super(HallOfFameRepair, self).__init__(excepted_size, *args, **kwargs)
        self.halloffame = halloffame
        self.top = top

    def generate_new(self, population, missing_count):
        ind = random.randint(0, min(self.top, len(self.halloffame)) - 1)
        return copy.deepcopy(population[ind])


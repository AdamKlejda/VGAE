import random
import copy

from evolalg.repair.repair import Repair


class MutateRepair(Repair):
    def __init__(self, mutate_step, excepted_size, iterations=1, *args, **kwargs):
        super(MutateRepair, self).__init__(excepted_size, *args, **kwargs)
        self.mutate_step = mutate_step
        self.iterations = iterations


    def generate_new(self, population, missing_count):
        selected = population[random.randint(0, len(population))]
        selected = copy.deepcopy(selected)
        for _ in range(self.iterations):
            selected = self.mutate_step([selected])[0]
        return selected

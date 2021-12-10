from evolalg.repair.repair import Repair
import copy

class MultistepRepair(Repair):
    def __init__(self, selection, steps, excepted_size, *args, **kwargs):
        super(MultistepRepair, self).__init__(excepted_size, *args, **kwargs)
        self.selection = selection
        self.steps = steps

    def generate_new(self, population, missing_count):
        selected = self.selection(population, missing_count)
        selected = copy.deepcopy(selected)
        selected = self.steps(selected)
        return selected

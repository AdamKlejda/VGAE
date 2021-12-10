import copy

from evolalg.repair.repair import Repair


class ConstRepair(Repair):
    def __init__(self, value, excepted_size, *args, **kwargs):
        super(ConstRepair, self).__init__(excepted_size, *args, **kwargs)
        self.value = value

    def generate_new(self, population, missing_count):
        return copy.deepcopy(self.value)

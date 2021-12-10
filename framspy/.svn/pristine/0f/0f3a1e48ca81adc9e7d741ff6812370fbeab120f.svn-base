from evolalg.statistics.halloffame_custom import HallOfFameCustom
from evolalg.statistics.statistics import Statistics


class HallOfFameStatistics(Statistics):
    def __init__(self, size, fields="fitness", *args, **kwargs):
        super(HallOfFameStatistics, self).__init__(*args, **kwargs)
        self.halloffame = HallOfFameCustom(size, fitness_field=fields)

    def init(self):
        self.halloffame.clear()

    def collect(self, population):
        self.halloffame.update(population)

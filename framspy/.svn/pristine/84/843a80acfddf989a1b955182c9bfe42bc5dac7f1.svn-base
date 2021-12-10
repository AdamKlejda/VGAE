from deap import tools

from evolalg.statistics.statistics import Statistics


class MultiStatistics(Statistics):
    def __init__(self, stats, verbose=True, *args, **kwargs):
        super(MultiStatistics, self).__init__(*args, **kwargs)
        self._org_stats = stats
        self.stats = tools.MultiStatistics(**stats)
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen'] + (self.stats.fields if self.stats else [])
        self.verbose = verbose
        self.gen = 0

    def init(self):
        for v in self._org_stats.values():
            v.init()
        self.gen = 0

    def collect(self, population):
        record = self.stats.compile(population)
        self.logbook.record(gen=self.gen, **record)
        self.gen += 1
        if self.verbose:
            print(self.logbook.stream)

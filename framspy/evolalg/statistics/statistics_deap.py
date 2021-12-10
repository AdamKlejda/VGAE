from deap import tools

from evolalg.statistics.statistics import Statistics


class StatisticsDeap(Statistics):
    def __init__(self, stats, extract_fn=lambda ind: ind.fitness, verbose=True, *args, **kwargs):
        super(StatisticsDeap, self).__init__(*args, **kwargs)
        self.stats = tools.Statistics(extract_fn)
        for name, fn in stats:
            self.stats.register(name, fn)

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen'] + (self.stats.fields if self.stats else [])

        self.gen = 0
        self.verbose = verbose

    def init(self):
        self.gen = 0

    def collect(self, population):
        record = self.stats.compile(population)
        self.logbook.record(gen=self.gen, **record)
        self.gen += 1
        if self.verbose:
            print(self.logbook.stream)

    def compile(self, data):
        return self.stats.compile(data)



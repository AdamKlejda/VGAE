from abc import abstractmethod

from evolalg.base.step import Step


class Statistics(Step):

    @abstractmethod
    def collect(self, population):
        pass

    def call(self, population):
        super(Statistics, self).call(population)
        self.collect(population)
        return population

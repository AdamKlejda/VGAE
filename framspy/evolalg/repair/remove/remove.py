from abc import abstractmethod
from evolalg.base.step import Step



class Remove(Step):
    def __init__(self, *args, **kwargs):
        super(Remove, self).__init__(*args , **kwargs)
        pass

    @abstractmethod
    def remove(self, individual):
        pass

    def call(self, population):
        super(Remove, self).call(population)
        return [_ for _ in population if not self.remove(_)]
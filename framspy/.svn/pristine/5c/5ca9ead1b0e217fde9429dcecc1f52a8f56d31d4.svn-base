import logging
from abc import abstractmethod


class Step:
    """
    Base abstract class for experiment's steps. It has three stages: pre, call and post.

    """

    def __init__(self, name=None):
        self.name = name
        if name is None:
            self.name = type(self).__name__


    def pre(self):
        pass

    @abstractmethod
    def call(self, population, *args, **kwargs):
        logging.getLogger(self.name).debug(f"Population size {len(population)}")

    def post(self):
        pass

    def init(self):
        pass

    def __call__(self, *args, **kwargs):
        self.pre()
        res = self.call(*args, **kwargs)
        self.post()
        return res

from deap.tools import HallOfFame
import operator
from copy import deepcopy
from bisect import bisect_right

class HallOfFameCustom(HallOfFame):
    def __init__(self,  maxsize, similar=operator.eq, fitness_field="fitness", *args, **kwargs):
        super(HallOfFameCustom, self).__init__(maxsize, similar, *args, **kwargs)
        print("HOF_size =",self.maxsize)
        self._fitness_field = fitness_field

    def extract_fitness(self, ind):
        return getattr(ind, self._fitness_field)

    def update(self, population):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.
        
        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """  
        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                for hofer in self:
                    if self.similar(ind, hofer):
                        break
                else:
                    self.insert(population[0])
                
                continue
            if self.extract_fitness(ind) >= self.extract_fitness(self[-1]) or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if self.similar(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better than
                    # the worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)

    def insert(self, item):
        """Insert a new individual in the hall of fame using the
        :func:`~bisect.bisect_right` function. The inserted individual is
        inserted on the right side of an equal individual. Inserting a new
        individual in the hall of fame also preserve the hall of fame's order.
        This method **does not** check for the size of the hall of fame, in a
        way that inserting a new individual in a full hall of fame will not
        remove the worst individual to maintain a constant size.

        :param item: The individual with a fitness attribute to insert in the
                     hall of fame.
        """
        item = deepcopy(item)
        i = bisect_right(self.keys, self.extract_fitness(item))
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, self.extract_fitness(item))


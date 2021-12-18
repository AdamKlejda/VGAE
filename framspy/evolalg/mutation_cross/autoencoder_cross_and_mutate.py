from evolalg.base.frams_step import FramsStep
from GAE.mutate_ae import AE_evolalg
import random
from evolalg.base.individual import Individual

class AutoencoderCrossAndMutate(FramsStep):
    def __init__(self, frams_lib, cross_prob, mutate_prob,path_config,train_id, m_range,mutate_commands=None, cross_commands=None, *args, **kwargs):
        super().__init__(frams_lib, *args, **kwargs)

        self.m_range = m_range
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.ae_mut = AE_evolalg(path_config,train_id)

    def call(self, population):
        super(AutoencoderCrossAndMutate, self).call(population)
        idx = []
        for i in range(len(population)):
            if random.random() < self.mutate_prob:
                idx.append(i)
        mutated = [population[_].genotype for _ in idx]
        idx, mutated = self.ae_mut.mutate_population(mutated,idx,m_range=self.m_range)
        
        for i, m in zip(idx, mutated):
            population[i] = Individual(m)

        return population

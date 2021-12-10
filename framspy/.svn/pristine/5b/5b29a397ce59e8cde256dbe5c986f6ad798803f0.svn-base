from abc import ABC

import numpy as np

from evolalg.base.frams_step import FramsStep
from evolalg.dissimilarity.dissimilarity import Dissimilarity

#TODO eliminate overlap with dissimilarity.py


class FramsDissimilarity(FramsStep):

    def __init__(self, frams_lib, reduction="mean", output_field="dissim", knn=None, *args, **kwargs):
        super(FramsDissimilarity, self).__init__(frams_lib, *args, **kwargs)

        self.output_field = output_field
        self.fn_reduce = None
        self.knn = knn
        if reduction == "mean":
            self.fn_reduce = np.mean
        elif reduction == "max":
            self.fn_reduce = np.max
        elif reduction == "min":
            self.fn_reduce = np.min
        elif reduction == "sum":
            self.fn_reduce = np.sum
        elif reduction == "knn_mean":
            self.fn_reduce = self.knn_mean
        elif reduction == "none" or reduction is None:
            self.fn_reduce = None
        else:
            raise ValueError("Unknown reduction type. Supported: mean, max, min, sum, knn_mean, none")

    def reduce(self, dissim_matrix):
        if self.fn_reduce is None:
            return dissim_matrix
        return self.fn_reduce(dissim_matrix, axis=1)

    def call(self, population):
        super(FramsDissimilarity, self).call(population)
        if len(population) == 0:
            return []
        dissim_matrix = self.frams.dissimilarity([_.genotype for _ in population])
        dissim = self.reduce(dissim_matrix)
        for d,ind in zip(dissim, population):
            setattr(ind, self.output_field, d)
        return population

    def knn_mean(self, dissim_matrix,axis):
        return np.mean(np.partition(dissim_matrix, self.knn)[:,:self.knn],axis=axis)
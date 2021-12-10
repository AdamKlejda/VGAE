from abc import ABC

from evolalg.base.step import Step
import numpy as np


class Dissimilarity(Step, ABC):

    def __init__(self, reduction="mean", output_field="dissim", knn=None, *args, **kwargs):
        super(Dissimilarity, self).__init__(*args, **kwargs)

        self.output_field = output_field
        self.fn_reduce = None
        self.knn = knn
        if reduction == "mean": # TODO change this 'elif' sequence to dictionary?
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

    def knn_mean(self, dissim_matrix,axis):
        return np.mean(np.partition(dissim_matrix, self.knn)[:,:self.knn],axis=axis)

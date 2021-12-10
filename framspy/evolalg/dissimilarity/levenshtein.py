import Levenshtein as lev

from evolalg.dissimilarity.dissimilarity import Dissimilarity


class LevenshteinDissimilarity(Dissimilarity):
    def __init__(self, reduction="mean", output_field="dissim", *args, **kwargs):
        super(LevenshteinDissimilarity, self).__init__(reduction, output_field, *args, **kwargs)

    def call(self, population):
        super(LevenshteinDissimilarity, self).call(population)
        if len(population) == 0:
            return []
        dissim = []
        for i, p in enumerate(population):
            gen_dis = []
            for i2, p2 in enumerate(population):
                gen_dis.append(lev.distance(p.genotype, p2.genotype))
            dissim.append(gen_dis)
        dissim = self.reduce(dissim)
        for d, ind in zip(dissim, population):
            setattr(ind, self.output_field, d)
        return population

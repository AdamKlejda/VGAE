from evolalg.base.step import Step


class LambdaStep(Step):
    """
    Wrapper for lambda expressions. In "normal" mode each step is given the entire population to work with. This class
    helps with any Callable that accepts individuals.

    """
    def __init__(self, fun, in_place=True, *args, **kwargs):
        """
        @param fun: Callable callable that will be called for each individual
        @param in_place: Bool
        """
        super(LambdaStep, self).__init__(*args, **kwargs)
        self.fun = fun
        self.in_place = in_place

    def call(self, population):
        super(LambdaStep, self).call(population)
        if self.in_place:
            [self.fun(_) for _ in population]
        else:
            population = [self.fun(_) for _ in population]
        return population

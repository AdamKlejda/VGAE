

from evolalg.repair.remove.remove import Remove
class LambdaRemove(Remove):
    def __init__(self, func, *args, **kwargs):
        super(LambdaRemove, self).__init__(*args, **kwargs)
        self.func = func

    def remove(self, individual):
        return self.func(individual)

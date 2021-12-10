from evolalg.base.step import Step
import os
from framsfiles import writer as framswriter


class PopulationSave(Step):
    def __init__(self, path, fields, provider=None, *args, **kwargs):
        super(PopulationSave, self).__init__(*args, **kwargs)
        self.path = path
        self.provider = provider
        self.fields = fields

    @staticmethod
    def ensure_dir(path):
        directory = os.path.dirname(path)
        if directory == "":
            return
        if not os.path.exists(directory):
            os.makedirs(directory)

    def call(self, population):
        super(PopulationSave, self).call(population)
        PopulationSave.ensure_dir(self.path)
        provider = self.provider
        if provider is None:
            provider = population

        #TODO instead of "fitness", write all fields used as fitness source with their original names (e.g. "velocity", "vertpos" etc.). In evaluation, set all attributes we get from Framsticks so that here we get all original names and values. Or, even better, introduce a dict field in Individual and assign to it everything that we get from Framsticks on evaluation (and add a filtering ability, i.e. when None - save all, when a list of field names - save only the enumerated fields)
        with open(self.path, "w") as outfile:
            for ind in provider:
                keyval = {}
                for k in self.fields: # construct a dictionary with criteria names and their values
                    keyval[k] = getattr(ind, self.fields[k])
                outfile.write(framswriter.from_collection({"_classname": "org", **keyval}))
                outfile.write("\n")

        print("Saved '%s' (%d)" % (self.path, len(provider)))
        return population

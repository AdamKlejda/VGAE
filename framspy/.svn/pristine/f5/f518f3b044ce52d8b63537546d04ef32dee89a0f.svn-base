from evolalg.base.step import Step


class FramsStep(Step):
    def __init__(self, frams_lib, commands=None, *args, **kwargs):
        super(FramsStep, self).__init__(*args, **kwargs)
        if commands is None:
            commands = []
        self.frams = frams_lib
        self.commands = commands

    def pre(self):
        for c in self.commands:
            self.frams.sendDirectCommand(c) #TODO update to use FramsticksLib when needed, pass lambda, import frams module? Maybe pre-lambda returns old value, post-lambda restores what pre-lambda returned. Or pass ExtValue or list of ExtValues and their new values (and either caller saves original values to be restored, or pre() returns a list of original values to be restored in post())

from typing import Iterable, Union, List

from evolalg.base.step import Step
import copy

def propagate_names(step: Step, prefix=""):

    if isinstance(step, Iterable):
        for s in step:
            if hasattr(step, "name"):
                propagate_names(s, prefix=prefix + "." + step.name)
            else:
                propagate_names(s, prefix=prefix)
    if hasattr(step, "name"):
        step.name = f"{prefix}.{step.name}"

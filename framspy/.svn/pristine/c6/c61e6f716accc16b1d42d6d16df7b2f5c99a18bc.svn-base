#TODO hof should be the complete non-dominated set from the entire process of evolution (all evaluated individuals), not limited in any way (remove '-hof_size'). Now we are not storing the entire Pareto front, but individuals that were better than others in hof [better=on all criteria?] at the time of inserting to hof. Hof has a size limit.
#TODO when -dissim is used, print its statistics just like all other criteria
#TODO in statistics, do not print "gen" (generation number) for each criterion
#TODO if possible, when saving the final .gen file, instead of fitness as a python tuple, save individual criteria - so instead of fitness:(0.005251036058321138, 0.025849976588613266), write "velocity:...\nvertpos:..." (this also applies to other .py examples in this directory)


import argparse
import logging
import os
import pickle
import sys
import copy
from enum import Enum

import numpy as np

from deap import base
from deap import tools

from FramsticksLib import FramsticksLib
from evolalg.base.lambda_step import LambdaStep
from evolalg.base.step import Step
from evolalg.dissimilarity.frams_dissimilarity import FramsDissimilarity
from evolalg.dissimilarity.levenshtein import LevenshteinDissimilarity
from evolalg.experiment import Experiment
from evolalg.fitness.fitness_step import FitnessStep
from evolalg.mutation_cross.frams_cross_and_mutate import FramsCrossAndMutate
from evolalg.population.frams_population import FramsPopulation
from evolalg.repair.remove.field import FieldRemove
from evolalg.repair.remove.remove import Remove
from evolalg.selection.nsga2 import NSGA2Selection
from evolalg.statistics.halloffame_stats import HallOfFameStatistics
from evolalg.statistics.multistatistics_deap import MultiStatistics
from evolalg.statistics.statistics_deap import StatisticsDeap
from evolalg.base.union_step import UnionStep
from evolalg.utils.population_save import PopulationSave


def ensureDir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


class Dissim(Enum):
    levenshtein = "levenshtein"
    frams = "frams"

    def __str__(self):
        return self.name



def parseArguments():
    parser = argparse.ArgumentParser(
        description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[
            0])
    parser.add_argument('-path', type=ensureDir, required=True,
                        help='Path to the Framsticks library without trailing slash.')
    parser.add_argument('-opt', required=True,
                        help='optimization criteria seperated with a comma: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, numconnections (and others as long as they are provided by the .sim file and its .expdef). Single or multiple criteria.')
    parser.add_argument('-lib', required=False, help="Filename of .so or .dll with the Framsticks library")

    parser.add_argument('-genformat', required=False, default="1",
                        help='Genetic format for the demo run, for example 4, 9, or B. If not given, f1 is assumed.')
    parser.add_argument('-sim', required=False, default="eval-allcriteria.sim",
                        help="Name of the .sim file with all parameter values. If you want to provide more files, separate them with a semicolon ';'.")
    parser.add_argument('-dissim', required=False, type=Dissim, default=Dissim.frams,
                        help='Dissimilarity measure, default: frams', choices=list(Dissim))
    parser.add_argument('-popsize', type=int, default=40, help="Population size (must be a multiple of 4), default: 40.") # mod 4 because of DEAP
    parser.add_argument('-generations', type=int, default=5, help="Number of generations, default: 5.")

    parser.add_argument('-max_numparts', type=int, default=None, help="Maximum number of Parts. Default: no limit")
    parser.add_argument('-max_numjoints', type=int, default=None, help="Maximum number of Joints. Default: no limit")
    parser.add_argument('-max_numneurons', type=int, default=None, help="Maximum number of Neurons. Default: no limit")
    parser.add_argument('-max_numconnections', type=int, default=None,
                        help="Maximum number of Neural connections. Default: no limit")

    parser.add_argument('-hof_size', type=int, default=10, help="Number of genotypes in Hall of Fame. Default: 10.")
    parser.add_argument('-hof_evaluations', type=int, default=20,
                        help="Number of final evaluations of each genotype in Hall of Fame to obtain reliable (averaged) fitness. Default: 20.")
    parser.add_argument('-checkpoint_path', required=False, default=None, help="Path to the checkpoint file")
    parser.add_argument('-checkpoint_interval', required=False, type=int, default=100, help="Checkpoint interval")
    parser.add_argument('-debug', dest='debug', action='store_true', help="Prints names of steps as they are executed")
    parser.set_defaults(debug=False)
    return parser.parse_args()


class NumPartsHigher(Remove):
    def __init__(self, max_number):
        super(NumPartsHigher, self).__init__()
        self.max_number = max_number

    def remove(self, individual):
        return individual.numparts > self.max_number


class NumJointsHigher(Remove):
    def __init__(self, max_number):
        super(NumJointsHigher, self).__init__()
        self.max_number = max_number

    def remove(self, individual):
        return individual.numjoints > self.max_number


class NumNeuronsHigher(Remove):
    def __init__(self, max_number):
        super(NumNeuronsHigher, self).__init__()
        self.max_number = max_number

    def remove(self, individual):
        return individual.numneurons > self.max_number


class NumConnectionsHigher(Remove):
    def __init__(self, max_number):
        super(NumConnectionsHigher, self).__init__()
        self.max_number = max_number

    def remove(self, individual):
        return individual.numconnections > self.max_number


class ReplaceWithHallOfFame(Step):
    def __init__(self, hof, *args, **kwargs):
        super(ReplaceWithHallOfFame, self).__init__(*args, **kwargs)
        self.hof = hof

    def call(self, population, *args, **kwargs):
        super(ReplaceWithHallOfFame, self).call(population)
        return list(self.hof.halloffame)


class DeapFitness(base.Fitness):
    weights = (1, 1)

    def __init__(self, *args, **kwargs):
        super(DeapFitness, self).__init__(*args, **kwargs)


class Nsga2Fitness:
    def __init__(self, fields):
        self.fields = fields
    def __call__(self, ind):
        setattr(ind, "fitness", DeapFitness(tuple(getattr(ind, _) for _ in self.fields)))



class ExtractField:
    def __init__(self, field_name):
        self.field_name = field_name
    def __call__(self, ind):
        return getattr(ind, self.field_name)

def extract_fitness(ind):
    return ind.fitness_raw


def load_experiment(path):
    with open(path, "rb") as file:
        experiment = pickle.load(file)
    print("Loaded experiment. Generation:", experiment.generation)
    return experiment


def create_experiment():
    parsed_args = parseArguments()
    frams_lib = FramsticksLib(parsed_args.path, parsed_args.lib, parsed_args.sim.split(";"))

    opt_dissim = []
    opt_fitness = []
    for crit in parsed_args.opt.split(','):
        try:
            Dissim(crit)
            opt_dissim.append(crit)
        except ValueError:
            opt_fitness.append(crit)
    if len(opt_dissim) > 1:
        raise ValueError("Only one type of dissimilarity supported")

    # Steps for generating first population
    init_stages = [
        FramsPopulation(frams_lib, parsed_args.genformat, parsed_args.popsize)
    ]

    # Selection procedure
    selection = NSGA2Selection(copy=True)

    # Procedure for generating new population. This steps will be run as long there is less than
    # popsize individuals in the new population
    new_generation_stages = [FramsCrossAndMutate(frams_lib, cross_prob=0.2, mutate_prob=0.9)]

    # Steps after new population is created. Executed exactly once per generation.
    generation_modifications = []

    # -------------------------------------------------
    # Fitness

    fitness_raw = FitnessStep(frams_lib, fields={**{_:_ for _ in opt_fitness},
                                                 "numparts": "numparts",
                                                 "numjoints": "numjoints",
                                                 "numneurons": "numneurons",
                                                 "numconnections": "numconnections"},
                              fields_defaults={parsed_args.opt: None, "numparts": float("inf"),
                                               "numjoints": float("inf"), "numneurons": float("inf"),
                                               "numconnections": float("inf"),
                                               **{_:None for _ in opt_fitness}
                                               },
                              evaluation_count=1)

    fitness_end = FitnessStep(frams_lib, fields={_:_ for _ in opt_fitness},
                              fields_defaults={parsed_args.opt: None},
                              evaluation_count=parsed_args.hof_evaluations)
    # Remove
    remove = []
    remove.append(FieldRemove(opt_fitness[0], None))  # Remove individuals if they have default value for fitness
    if parsed_args.max_numparts is not None:
        # This could be also implemented by "LambdaRemove(lambda x: x.numparts > parsed_args.num_parts)"
        # But this would not serialize in checkpoint.
        remove.append(NumPartsHigher(parsed_args.max_numparts))
    if parsed_args.max_numjoints is not None:
        remove.append(NumJointsHigher(parsed_args.max_numjoints))
    if parsed_args.max_numneurons is not None:
        remove.append(NumNeuronsHigher(parsed_args.max_numneurons))
    if parsed_args.max_numconnections is not None:
        remove.append(NumConnectionsHigher(parsed_args.max_numconnections))

    remove_step = UnionStep(remove)

    fitness_remove = UnionStep([fitness_raw, remove_step])

    init_stages.append(fitness_remove)
    new_generation_stages.append(fitness_remove)

    # -------------------------------------------------
    # Dissimilarity as one of the criteria
    dissim = None
    if len(opt_dissim) > 0 and Dissim(opt_dissim[0]) == Dissim.levenshtein:
        dissim = LevenshteinDissimilarity(reduction="mean", output_field="dissim")
    elif len(opt_dissim) > 0 and Dissim(opt_dissim[0]) == Dissim.frams:
        dissim = FramsDissimilarity(frams_lib, reduction="mean", output_field="dissim")

    if dissim is not None:
        init_stages.append(dissim)
        generation_modifications.append(dissim)

    if dissim is not None:
        nsga2_fittnes = Nsga2Fitness(["dissim"]+ opt_fitness)
    else:
        nsga2_fittnes = Nsga2Fitness(opt_fitness)

    init_stages.append(LambdaStep(nsga2_fittnes))
    generation_modifications.append(LambdaStep(nsga2_fittnes))

    # -------------------------------------------------
    # Statistics
    hall_of_fame = HallOfFameStatistics(parsed_args.hof_size, "fitness")  # Wrapper for halloffamae
    replace_with_hof = ReplaceWithHallOfFame(hall_of_fame)
    statistics_deap = MultiStatistics({fit:StatisticsDeap([
        ("avg", np.mean),
        ("stddev", np.std),
        ("min", np.min),
        ("max", np.max)
    ], ExtractField(fit)) for fit in opt_fitness}) # Wrapper for deap statistics

    statistics_union = UnionStep(
        [hall_of_fame,
        statistics_deap]
    )  # Union of two statistics steps.

    init_stages.append(statistics_union)
    generation_modifications.append(statistics_union)

    # -------------------------------------------------
    # End stages: this will execute exactly once after all generations.
    end_stages = [
        replace_with_hof,
        fitness_end,
        PopulationSave("halloffame.gen", provider=hall_of_fame.halloffame, fields={"genotype": "genotype",
                                                                                   "fitness": "fitness"})]
    # ...but custom fields can be added, e.g. "custom": "recording"

    # -------------------------------------------------

    # Experiment creation

    experiment = Experiment(init_population=init_stages,
                            selection=selection,
                            new_generation_steps=new_generation_stages,
                            generation_modification=generation_modifications,
                            end_steps=end_stages,
                            population_size=parsed_args.popsize,
                            checkpoint_path=parsed_args.checkpoint_path,
                            checkpoint_interval=parsed_args.checkpoint_interval
                            )
    return experiment


def main():
    print("Running experiment with", sys.argv)
    parsed_args = parseArguments()
    
    if parsed_args.popsize % 4 != 0:
    	raise ValueError("popsize must be a multiple of 4 (for example %d)." % (parsed_args.popsize//4*4)) # required by deap.tools.selTournamentDCD() 
    	
    if parsed_args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if parsed_args.checkpoint_path is not None and os.path.exists(parsed_args.checkpoint_path):
        experiment = load_experiment(parsed_args.checkpoint_path)
    else:
        experiment = create_experiment()
        experiment.init()  # init is mandatory

    experiment.run(parsed_args.generations)

    # Next call for experiment.run(10) will do nothing. Parameter 10 specifies how many generations should be
    # in one experiment. Previous call generated 10 generations.
    # Example 1:
    # experiment.init()
    # experiment.run(10)
    # experiment.run(12)
    # #This will run for total of 12 generations
    #
    # Example 2
    # experiment.init()
    # experiment.run(10)
    # experiment.init()
    # experiment.run(10)
    # # All work produced by first run will be "destroyed" by second init().


if __name__ == '__main__':
    main()

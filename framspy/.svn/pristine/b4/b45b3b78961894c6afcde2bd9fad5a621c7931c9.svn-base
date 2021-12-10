import argparse
import logging
import os
import pickle
import sys
from enum import Enum

import numpy as np

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
from evolalg.selection.tournament import TournamentSelection
from evolalg.statistics.halloffame_stats import HallOfFameStatistics
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


class Fitness(Enum):
    raw = "raw"
    niching = "niching"
    novelty = "novelty"
    knn_niching = "knn_niching"
    knn_novelty = "knn_novelty"

    def __str__(self):
        return self.name


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[
            0])
    parser.add_argument('-path', type=ensureDir, required=True, help='Path to the Framsticks library without trailing slash.')
    parser.add_argument('-opt', required=True,
                        help='optimization criteria: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, numconnections (or other as long as it is provided by the .sim file and its .expdef). For multiple criteria optimization, see multicriteria.py.')
    parser.add_argument('-lib', required=False, help="Filename of .so or .dll with the Framsticks library")

    parser.add_argument('-genformat', required=False, default="1",
                        help='Genetic format for the demo run, for example 4, 9, or B. If not given, f1 is assumed.')
    parser.add_argument('-sim', required=False, default="eval-allcriteria.sim", help="Name of the .sim file with all parameter values. If you want to provide more files, separate them with a semicolon ';'.")
    parser.add_argument('-fit', required=False, default=Fitness.raw, type=Fitness,
                        help=' Fitness criteria, default: raw', choices=list(Fitness))
    parser.add_argument('-dissim', required=False, type=Dissim, default=Dissim.frams,
                        help='Dissimilarity measure, default: frams', choices=list(Dissim))
    parser.add_argument('-knn', type=int, help="'k' value for knn-based fitness criteria (knn-niching and knn-novelty).")
    parser.add_argument('-popsize', type=int, default=50, help="Population size, default: 50.")
    parser.add_argument('-generations', type=int, default=5, help="Number of generations, default: 5.")
    parser.add_argument('-tournament', type=int, default=5, help="Tournament size, default: 5.")

    parser.add_argument('-max_numparts', type=int, default=None, help="Maximum number of Parts. Default: no limit")
    parser.add_argument('-max_numjoints', type=int, default=None, help="Maximum number of Joints. Default: no limit")
    parser.add_argument('-max_numneurons', type=int, default=None, help="Maximum number of Neurons. Default: no limit")
    parser.add_argument('-max_numconnections', type=int, default=None, help="Maximum number of Neural connections. Default: no limit")

    parser.add_argument('-hof_size', type=int, default=10, help="Number of genotypes in Hall of Fame. Default: 10.")
    parser.add_argument('-hof_evaluations', type=int, default=20, help="Number of final evaluations of each genotype in Hall of Fame to obtain reliable (averaged) fitness. Default: 20.")
    parser.add_argument('-checkpoint_path', required=False, default=None, help="Path to the checkpoint file")
    parser.add_argument('-checkpoint_interval', required=False, type=int, default=100, help="Checkpoint interval")
    parser.add_argument('-debug', dest='debug', action='store_true', help="Prints names of steps as they are executed")
    parser.set_defaults(debug=False)
    return parser.parse_args()


def extract_fitness(ind):
    return ind.fitness_raw


def print_population_count(pop):
    print("Current popsize:", len(pop))
    return pop  # Each step must return a population


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


def func_raw(ind): setattr(ind, "fitness", ind.fitness_raw)


def func_novelty(ind): setattr(ind, "fitness", ind.dissim)


def func_knn_novelty(ind): setattr(ind, "fitness", ind.dissim)


def func_niching(ind): setattr(ind, "fitness", ind.fitness_raw * (1 + ind.dissim))


def func_knn_niching(ind): setattr(ind, "fitness", ind.fitness_raw * (1 + ind.dissim))


def load_experiment(path):
    with open(path, "rb") as file:
        experiment = pickle.load(file)
    print("Loaded experiment. Generation:", experiment.generation)
    return experiment


def create_experiment():
    parsed_args = parseArguments()
    frams_lib = FramsticksLib(parsed_args.path, parsed_args.lib,
                          parsed_args.sim.split(";"))
    # Steps for generating first population
    init_stages = [
        FramsPopulation(frams_lib, parsed_args.genformat, parsed_args.popsize)
    ]

    # Selection procedure
    selection = TournamentSelection(parsed_args.tournament,
                                    copy=True)  # 'fitness' by default, the targeted attribute can be changed, e.g. fit_attr="fitness_raw"

    # Procedure for generating new population. This steps will be run as long there is less than
    # popsize individuals in the new population
    new_generation_stages = [FramsCrossAndMutate(frams_lib, cross_prob=0.2, mutate_prob=0.9)]

    # Steps after new population is created. Executed exactly once per generation.
    generation_modifications = []

    # -------------------------------------------------
    # Fitness

    fitness_raw = FitnessStep(frams_lib, fields={parsed_args.opt: "fitness_raw",
                                             "numparts": "numparts",
                                             "numjoints": "numjoints",
                                             "numneurons": "numneurons",
                                             "numconnections": "numconnections"},
                              fields_defaults={parsed_args.opt: None, "numparts": float("inf"),
                                               "numjoints": float("inf"), "numneurons": float("inf"),
                                               "numconnections": float("inf")},
                              evaluation_count=1)


    fitness_end = FitnessStep(frams_lib, fields={parsed_args.opt: "fitness_raw"},
                              fields_defaults={parsed_args.opt: None},
                              evaluation_count=parsed_args.hof_evaluations)
    # Remove
    remove = []
    remove.append(FieldRemove("fitness_raw", None))  # Remove individuals if they have default value for fitness
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
    # Novelty or niching
    knn = parsed_args.knn
    if parsed_args.fit == Fitness.knn_novelty or parsed_args.fit == Fitness.knn_niching:
        reduction_method = "knn_mean"
        assert knn is not None, "'k' must be set for knn-based fitness."
        assert knn > 0, "'k' must be positive."
        assert knn < parsed_args.popsize, "'k' must be smaller than population size."
    else:
        reduction_method = "mean"
        assert knn is None, "'k' is irrelevant unless knn-based fitness is used."

    dissim = None
    if parsed_args.dissim == Dissim.levenshtein:
        dissim = LevenshteinDissimilarity(reduction=reduction_method, knn=knn, output_field="dissim")
    elif parsed_args.dissim == Dissim.frams:
        dissim = FramsDissimilarity(frams_lib, reduction=reduction_method, knn=knn, output_field="dissim")

    if parsed_args.fit == Fitness.raw:
        # Fitness is equal to finess raw
        raw = LambdaStep(func_raw)
        init_stages.append(raw)
        generation_modifications.append(raw)

    if parsed_args.fit == Fitness.niching: # TODO reduce redundancy in the four cases below: dictionary?
        niching = UnionStep([
            dissim,
            LambdaStep(func_niching)
        ])
        init_stages.append(niching)
        generation_modifications.append(niching)

    if parsed_args.fit == Fitness.novelty:
        novelty = UnionStep([
            dissim,
            LambdaStep(func_novelty)
        ])
        init_stages.append(novelty)
        generation_modifications.append(novelty)
    
    if parsed_args.fit == Fitness.knn_niching:
        knn_niching = UnionStep([
            dissim,
            LambdaStep(func_knn_niching)
        ])
        init_stages.append(knn_niching)
        generation_modifications.append(knn_niching)
    
    if parsed_args.fit == Fitness.knn_novelty:
        knn_novelty = UnionStep([
            dissim,
            LambdaStep(func_knn_novelty)
        ])
        init_stages.append(knn_novelty)
        generation_modifications.append(knn_novelty)

    # -------------------------------------------------
    # Statistics
    hall_of_fame = HallOfFameStatistics(parsed_args.hof_size, "fitness_raw")  # Wrapper for halloffamae
    replace_with_hof = ReplaceWithHallOfFame(hall_of_fame)
    statistics_deap = StatisticsDeap([
        ("avg", np.mean),
        ("stddev", np.std),
        ("min", np.min),
        ("max", np.max)
    ], extract_fitness)  # Wrapper for deap statistics

    statistics_union = UnionStep([
        hall_of_fame,
        statistics_deap
    ])  # Union of two statistics steps.

    init_stages.append(statistics_union)
    generation_modifications.append(statistics_union)

    # -------------------------------------------------
    # End stages: this will execute exactly once after all generations.
    end_stages = [
        replace_with_hof,
        fitness_end,
        PopulationSave("halloffame.gen", provider=hall_of_fame.halloffame, fields={"genotype": "genotype",
                                                                                  "fitness": "fitness_raw"})]
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

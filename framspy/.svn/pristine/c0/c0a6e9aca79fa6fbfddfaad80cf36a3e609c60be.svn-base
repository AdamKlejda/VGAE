import argparse
import os
import sys
import numpy as np
from deap import creator, base, tools, algorithms
from FramsticksLib import FramsticksLib

# Note: this may be less efficient than running the evolution directly in Framsticks, so if performance is key, compare both options.


# The list of criteria includes 'vertpos', 'velocity', 'distance', 'vertvel', 'lifespan', 'numjoints', 'numparts', 'numneurons', 'numconnections'.
OPTIMIZATION_CRITERIA = ['velocity']  # Single or multiple criteria. Names from the standard-eval.expdef dictionary, e.g. ['vertpos', 'velocity'].


def frams_evaluate(frams_cli, individual):
	genotype = individual[0]  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
	data = frams_cli.evaluate([genotype])
	# print("Evaluated '%s'" % genotype, 'evaluation is:', data)
	try:
		first_genotype_data = data[0]
		evaluation_data = first_genotype_data["evaluations"]
		default_evaluation_data = evaluation_data[""]
		fitness = [default_evaluation_data[crit] for crit in OPTIMIZATION_CRITERIA]
	except (KeyError, TypeError) as e:  # the evaluation may have failed for an invalid genotype (such as X[@][@] with "Don't simulate genotypes with warnings" option) or for some other reason
		fitness = [-1] * len(OPTIMIZATION_CRITERIA)  # fitness of -1 is intended to discourage further propagation of this genotype via selection ("this one is very poor")
		print('Error "%s": could not evaluate genotype "%s", returning fitness %s' % (str(e), genotype, fitness))
	return fitness


def frams_crossover(frams_cli, individual1, individual2):
	geno1 = individual1[0]  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
	geno2 = individual2[0]  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
	individual1[0] = frams_cli.crossOver(geno1, geno2)
	individual2[0] = frams_cli.crossOver(geno1, geno2)
	return individual1, individual2


def frams_mutate(frams_cli, individual):
	individual[0] = frams_cli.mutate([individual[0]])[0]  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
	return individual,


def frams_getsimplest(frams_cli, genetic_format):
	return frams_cli.getSimplest(genetic_format)


def prepareToolbox(frams_cli, genetic_format):
	creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OPTIMIZATION_CRITERIA))
	creator.create("Individual", list, fitness=creator.FitnessMax)  # would be nice to have "str" instead of unnecessary "list of str"

	toolbox = base.Toolbox()
	toolbox.register("attr_simplest_genotype", frams_getsimplest, frams_cli, genetic_format)  # "Attribute generator"
	# (failed) struggle to have an individual which is a simple str, not a list of str
	# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_frams)
	# https://stackoverflow.com/questions/51451815/python-deap-library-using-random-words-as-individuals
	# https://github.com/DEAP/deap/issues/339
	# https://gitlab.com/santiagoandre/deap-customize-population-example/-/blob/master/AGbasic.py
	# https://groups.google.com/forum/#!topic/deap-users/22g1kyrpKy8
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("evaluate", frams_evaluate, frams_cli)
	toolbox.register("mate", frams_crossover, frams_cli)
	toolbox.register("mutate", frams_mutate, frams_cli)
	if len(OPTIMIZATION_CRITERIA) <= 1:
		toolbox.register("select", tools.selTournament, tournsize=5)
	else:
		toolbox.register("select", tools.selNSGA2)
	return toolbox


def parseArguments():
	parser = argparse.ArgumentParser(description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
	parser.add_argument('-path', type=ensureDir, required=True, help='Path to Framsticks CLI without trailing slash.')
	parser.add_argument('-lib', required=False, help='Library name. If not given, "frams-objects.dll" or "frams-objects.so" is assumed depending on the platform.')
	parser.add_argument('-simsettings', required=False, help='The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If not given, "eval-allcriteria.sim" is assumed by default. Must be compatible with the "standard-eval" expdef.')
	parser.add_argument('-genformat', required=False, help='Genetic format for the demo run, for example 4, 9, or B. If not given, f1 is assumed.')
	return parser.parse_args()


def ensureDir(string):
	if os.path.isdir(string):
		return string
	else:
		raise NotADirectoryError(string)


if __name__ == "__main__":
	# A demo run: optimize OPTIMIZATION_CRITERIA

	# random.seed(123)  # see FramsticksLib.DETERMINISTIC below, set to True if you want full determinism
	FramsticksLib.DETERMINISTIC = False  # must be set before FramsticksLib() constructor call
	parsed_args = parseArguments()
	framsLib = FramsticksLib(parsed_args.path, parsed_args.lib, parsed_args.simsettings)

	toolbox = prepareToolbox(framsLib, '1' if parsed_args.genformat is None else parsed_args.genformat)

	POPSIZE = 20
	GENERATIONS = 50

	pop = toolbox.population(n=POPSIZE)
	hof = tools.HallOfFame(5)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("stddev", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	print('Evolution with population size %d for %d generations, optimization criteria: %s' % (POPSIZE, GENERATIONS, OPTIMIZATION_CRITERIA))
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.9, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)
	print('Best individuals:')
	for best in hof:
		print(best.fitness, '\t-->\t', best[0])

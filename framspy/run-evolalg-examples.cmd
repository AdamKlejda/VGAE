rem To learn about all available options of each .py algorithm below, add "-h" to its parameters.
rem Use the source code of the examples as a starting point for your customizations.
rem Example usage:

set DIR_WITH_FRAMS_LIBRARY=............



rem simple one-criterion evolution
python -m evolalg.examples.standard          -path %DIR_WITH_FRAMS_LIBRARY%   -opt numneurons


rem "chaining" .sim files, subsequent files overwrite selected parameters
python -m evolalg.examples.standard          -path %DIR_WITH_FRAMS_LIBRARY%   -sim eval-allcriteria.sim;deterministic.sim;sample-period-longest.sim    -opt velocity


rem hard limit on the number of parts
python -m evolalg.examples.niching_novelty   -path %DIR_WITH_FRAMS_LIBRARY%   -opt velocity   -max_numparts 6   -debug


rem "local" niching
python -m evolalg.examples.niching_novelty   -path %DIR_WITH_FRAMS_LIBRARY%   -opt vertpos    -fit knn_niching  -knn 3    -max_numjoints 8 -popsize 10 -generations 30


rem two criteria, '-dissim ...' can also be used to include dissimilarity as one of the criteria
python -m evolalg.examples.multicriteria     -path %DIR_WITH_FRAMS_LIBRARY%   -popsize 40 -generations 10 -opt velocity,vertpos 

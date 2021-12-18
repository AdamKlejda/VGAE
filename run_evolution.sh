#!/bin/bash
PATH_FRAMS=""
OPT="vertpos"
PATH_CONFIG="../$1"
RANGE=0.333
cd framspy
srun -p idss-student  python3 -u -m  evolalg.examples.standard_gae -path $PATH_FRAMS -opt $OPT -path_config $PATH_CONFIG -m_range $RANGE -train_id $SLURM_ARRAY_TASK_ID -id 0
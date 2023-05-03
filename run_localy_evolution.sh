#!/bin/bash
PATH_FRAMS="/home/adam/Framsticks/Framsticks50rc19/"
OPT="vertpos"
PATH_CONFIG=$1
RANGE=0.333
cd framspy
python3 -u -m  evolalg.examples.standard_gae -path $PATH_FRAMS -opt $OPT -path_config $PATH_CONFIG -m_range $RANGE -id 1 -train_id 1 #$SLURM_ARRAY_TASK_ID
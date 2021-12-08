#!/bin/bash
#SBATCH --time=24:00:00
PATH_FRAMS=$1
PATH_DATA=$2
PATH_OUT=$3
BATCH_SIZE=$4
ADJ_SIZE=$5
NUM_FEATURES=$6
LATENT_DIM=$7
NHIDDEN=$8
CONVENC=$9
DENSEENC=${10}
DENSEDECA=${11}
CONVDECX=${12}
DENSEDECX=${13}
LEARNING_RATE=${14}
EPOCHS=${15}
CONVTYPE=${16}
VARIATIONAL=${17} 
LOSS=${18} 

cd framspy

srun -p idss-student python3 -u -m GAE.training.py -pathframs $PATH_FRAMS -pathdata $PATH_DATA -pathout $PATH_OUT -batchsize $BATCH_SIZE -adjsize $ADJ_SIZE -numfeatures $NUM_FEATURES -latentdim $LATENT_DIM -nhidden $NHIDDEN -convtype $CONVTYPE -convenc $CONVENC -denseenc $DENSEENC -densedeca $DENSEDECA -convdecx $CONVDECX -densedecx $DENSEDECX -learningrate $LEARNING_RATE -epochs $EPOCHS -variational $VARIATIONAL -loss $LOSS -trainid $SLURM_ARRAY_TASK_ID
#!/bin/bash
PATH_FRAMS=/home/adam/Framsticks/Framsticks50rc19
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
# TRAINID=${17} 
VARIATIONAL=${17}
# echo $VARIATIONAL


python3 -u -m training_test.py -pathframs $PATH_FRAMS -pathdata $PATH_DATA -pathout $PATH_OUT -batchsize $BATCH_SIZE -adjsize $ADJ_SIZE -numfeatures $NUM_FEATURES -latentdim $LATENT_DIM -nhidden $NHIDDEN -convtype $CONVTYPE -convenc $CONVENC -denseenc $DENSEENC -densedeca $DENSEDECA -convdecx $CONVDECX -densedecx $DENSEDECX -learningrate $LEARNING_RATE -epochs $EPOCHS -trainid 1 -variational $VARIATIONAL
# python3 -u -m training_l.py -pathframs $PATH_FRAMS -pathdata $PATH_DATA -pathout $PATH_OUT -batchsize $BATCH_SIZE -adjsize $ADJ_SIZE -numfeatures $NUM_FEATURES -latentdim $LATENT_DIM -nhidden $NHIDDEN -convtype $CONVTYPE -convenc $CONVENC -denseenc $DENSEENC -densedeca $DENSEDECA -convdecx $CONVDECX -densedecx $DENSEDECX -learningrate $LEARNING_RATE -epochs $EPOCHS -trainid $TRAINID -variational $VARIATIONAL

# python3 -u -m training.py -pathframs $PATH_FRAMS -pathdata $PATH_DATA -pathout $PATH_OUT -batchsize $BATCH_SIZE -adjsize $ADJ_SIZE -numfeatures $NUM_FEATURES -latentdim $LATENT_DIM -nhidden $NHIDDEN -convtype $CONVTYPE -convenc $CONVENC -denseenc $DENSEENC -densedeca $DENSEDECA -convdecx $CONVDECX -densedecx $DENSEDECX -learningrate $LEARNING_RATE -epochs $EPOCHS -trainid 1 -variational $VARIATIONAL


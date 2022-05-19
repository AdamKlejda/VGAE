#!/bin/bash
PATH_FRAMS=/home/adam/Framsticks/Framsticks50rc19
PATH_DATA=$2
PATH_OUT=$3 #/home/adam/thesis/VGAE/experiments_new/configs/models/
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
TRAINID=1 

if [[ $LOSS =~ No ]] 
then
   LOSS=None
fi

# echo $VARIATIONAL
cd framspy

# python3 -u -m   GAE.training_test.py -pathframs $PATH_FRAMS -pathdata $PATH_DATA -pathout $PATH_OUT -batchsize $BATCH_SIZE -adjsize $ADJ_SIZE -numfeatures $NUM_FEATURES -latentdim $LATENT_DIM -nhidden $NHIDDEN -convtype $CONVTYPE -convenc $CONVENC -denseenc $DENSEENC -densedeca $DENSEDECA -convdecx $CONVDECX -densedecx $DENSEDECX -learningrate $LEARNING_RATE -epochs $EPOCHS -variational $VARIATIONAL -loss $LOSS  -trainid $TRAINID
# python3 -u -m   mutate_ae.py -pathframs $PATH_FRAMS -pathdata $PATH_DATA -pathout $PATH_OUT -batchsize $BATCH_SIZE -adjsize $ADJ_SIZE -numfeatures $NUM_FEATURES -latentdim $LATENT_DIM -nhidden $NHIDDEN -convtype $CONVTYPE -convenc $CONVENC -denseenc $DENSEENC -densedeca $DENSEDECA -convdecx $CONVDECX -densedecx $DENSEDECX -learningrate $LEARNING_RATE -epochs $EPOCHS -variational $VARIATIONAL -loss $LOSS  -trainid $TRAINID
# python3 -u -m  mutate_ae.py -pathconfig '/home/adam/thesis/VGAE/experiments/experiment_base/models/No/GAE/numfeatures3/adjsize15/batchsize256/latentdim15/nhidden64/learningrate0.01/convtypeGCSConv/model_enc_2_2_deca1_decx1_2_train_id_9/args.txt'
python3 -u -m GAE.training.py -pathframs $PATH_FRAMS -pathdata $PATH_DATA -pathout $PATH_OUT -batchsize $BATCH_SIZE -adjsize $ADJ_SIZE -numfeatures $NUM_FEATURES -latentdim $LATENT_DIM -nhidden $NHIDDEN -convtype $CONVTYPE -convenc $CONVENC -denseenc $DENSEENC -densedeca $DENSEDECA -convdecx $CONVDECX -densedecx $DENSEDECX -learningrate $LEARNING_RATE -epochs $EPOCHS -variational $VARIATIONAL -loss $LOSS  -trainid 1


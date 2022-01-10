import spektral.data as data
import tensorflow as tf
import numpy as np 
from tensorflow import keras
import os
import sys
import argparse
import math
from math import floor

from GAE.GraphDataset import GraphDataset
from GAE.autoencoder import EncoderGAE,EncoderVGAE, DecoderX, DecoderA, VGAE, GAE, Weights
from GAE.utils import *
from GAE.custom_layers import ConvTypes
from GAE.LossManager import LossManager, LossTypes

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def ensureDir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
        
def save_config(parsed_args,path_args):    
    with open(path_args, "w") as file:
        file.write(str(parsed_args.pathframs)+"\n")    #pathframs
        file.write(str(parsed_args.pathdata)+"\n")     #pathdata
        file.write(str(parsed_args.pathout)+"\n")      #pathout
        file.write(str(parsed_args.batchsize)+"\n")    #batchsize
        file.write(str(parsed_args.adjsize)+"\n")      #adjsize
        file.write(str(parsed_args.numfeatures)+"\n")  #numfeatures
        file.write(str(parsed_args.latentdim)+"\n")     #latentdim
        file.write(str(parsed_args.nhidden)+"\n")       #nhidden
        file.write(str(parsed_args.convenc)+"\n")       #convenc
        file.write(str(parsed_args.denseenc)+"\n")      #denseenc
        file.write(str(parsed_args.densedeca)+"\n")     #densedeca
        file.write(str(parsed_args.convdecx)+"\n")      #convdecx
        file.write(str(parsed_args.densedecx)+"\n")     #densedecx
        file.write(str(parsed_args.learningrate)+"\n") #learningrate
        file.write(str(parsed_args.epochs)+"\n")        #epochs
        file.write(str(parsed_args.convtype.value)+"\n")#convtype
        file.write(str(parsed_args.variational)+"\n")   #variational
        file.write(str(parsed_args.loss)+"\n")          #loss
        file.write(str(parsed_args.trainid)+"\n")       #trainid


def test_model(autoencoder,data_loader,steps_test,variational):
    loss_all =[]
    for _ in range(steps_test):
        (x,a),y= next(data_loader)
        if variational:
            total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX,reconstruction_lossMask,kl_loss = autoencoder.test_step([(tf.convert_to_tensor(x)),
                                    tf.convert_to_tensor(a),
                                    y])
            loss_all.append([total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX,reconstruction_lossMask,kl_loss])

        else:
            total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX,reconstruction_lossMask = autoencoder.test_step([(tf.convert_to_tensor(x)),
                                    tf.convert_to_tensor(a),
                                    y])
            loss_all.append([total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX,reconstruction_lossMask])
    return np.array(loss_all).mean(axis=0)

def parseArguments():
    parser = argparse.ArgumentParser(
        description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[
            0])
    parser.add_argument('-pathframs', type=ensureDir, required=True, help='Path to the Framsticks library without trailing slash.')
    parser.add_argument('-pathdata', type=ensureDir, required=True, help='Path to the data input in gen format.')
    parser.add_argument('-pathout', type=ensureDir, required=True, help='Path to save output model and plots.')
    parser.add_argument('-batchsize', type=int, required=False, help='',default=256)
    parser.add_argument('-adjsize', type=int, required=False, help='',default=15)
    parser.add_argument('-numfeatures', type=int, required=False, help='',default=4)
    parser.add_argument('-latentdim', type=int, required=False, help='',default=3)
    parser.add_argument('-nhidden', type=int,   required=False, help='',default=16)
    parser.add_argument('-convtype', type=ConvTypes, required=False, help='',default=ConvTypes.GCNConv)
    parser.add_argument('-convenc', type=int, required=False, help='',default=0)
    parser.add_argument('-denseenc', type=int, required=False, help='',default=1)
    parser.add_argument('-densedeca', type=int, required=False, help='',default=1)
    parser.add_argument('-convdecx', type=int, required=False, help='',default=1)
    parser.add_argument('-densedecx', type=int, required=False, help='',default=1)
    parser.add_argument('-learningrate', type=float, required=False, help='',default=0.0001)
    parser.add_argument('-epochs', type=int, required=False, help='',default=500)
    parser.add_argument('-trainid', type=int, required=False, help='',default=-1)
    parser.add_argument('-variational', type=str, required=True, help='')
    parser.add_argument('-loss', type=LossTypes, required=True, help='',default=LossTypes.No)
    parser.set_defaults(debug=False)
    return parser.parse_args()
ae_type=None
parsed_args = parseArguments()

print(parsed_args.__dict__)

if parsed_args.variational == "True":
    variational=True
else:
    variational=False

if variational == True:
    ae_type="VGAE"
else:
    ae_type="GAE"


PATH_OUT = (str(parsed_args.pathout)+
            str(parsed_args.loss)+
            "/"+ae_type +
            "/numfeatures"+str(parsed_args.numfeatures) +
            "/adjsize"+str(parsed_args.adjsize) + 
            "/batchsize"+str(parsed_args.batchsize) +
            "/latentdim"+str(parsed_args.latentdim)+
            "/nhidden"+str(parsed_args.nhidden)+
            "/learningrate"+str(parsed_args.learningrate)+
            "/convtype"+str(parsed_args.convtype)+
            "/"
            )
MODEL_NAME = ("model_enc_"+str(parsed_args.convenc)+"_"+str(parsed_args.denseenc)+
             "_deca"+str(parsed_args.densedeca)+
             "_decx"+str(parsed_args.convdecx)+"_"+str(parsed_args.densedecx)+
             "_train_id_"+str(parsed_args.trainid)
             )

fitness = None

if parsed_args.loss is not LossTypes.No:
    lossManager = LossManager(parsed_args.pathframs,"eval-allcriteria_new.sim","vertpos")
    if parsed_args.loss == LossTypes.joints:
        custom_loss = lossManager.joints_too_big_loss
    elif parsed_args.loss == LossTypes.parts:
        custom_loss = lossManager.part_number_loss
    elif parsed_args.loss == LossTypes.fitness:
        fitness="vertpos"
        custom_loss = lossManager.fitness_comparison_loss
    elif parsed_args.loss == LossTypes.dissim:
        custom_loss = lossManager.dissimilarity_comparison
    else:
        print(parsed_args.loss," is not supported, custom loss set to None")
        custom_loss = None
else:
    custom_loss = None

train, test = GraphDataset(parsed_args.pathframs, parsed_args.pathdata,max_examples=999999,fitness=fitness,size_of_adj=parsed_args.adjsize).read()
# train, test = GraphDataset(parsed_args.pathframs, parsed_args.pathdata,max_examples=500,fitness=fitness,size_of_adj=parsed_args.adjsize).read()

loader_train = data.BatchLoader(train, batch_size=parsed_args.batchsize)
loader_test = data.BatchLoader(test, batch_size=parsed_args.batchsize)

if variational == True:
    encoder = EncoderVGAE(latent_dim=parsed_args.latentdim,
                    n_hidden=parsed_args.nhidden,
                    num_conv=parsed_args.convenc,
                    num_dense=parsed_args.denseenc,
                    convtype=parsed_args.convtype)
else:
    encoder = EncoderGAE(latent_dim=parsed_args.latentdim,
                  n_hidden=parsed_args.nhidden,
                  num_conv=parsed_args.convenc,
                  num_dense=parsed_args.denseenc,
                  convtype=parsed_args.convtype)
                    
decoderA = DecoderA(adjency_size=parsed_args.adjsize,
                    latent_dim=parsed_args.latentdim,
                    num_dense=parsed_args.densedeca)
decoderX = DecoderX(latent_dim=parsed_args.latentdim,
                    adjency_size=parsed_args.adjsize,
                    num_features=parsed_args.numfeatures,
                    num_conv=parsed_args.convdecx,
                    num_dense=parsed_args.densedecx,
                    convtype=parsed_args.convtype)


current_learning_rate = parsed_args.learningrate

if variational == True:
    autoencoder = VGAE(encoder,decoderA,decoderX,custom_loss)
else:
    autoencoder = GAE(encoder,decoderA,decoderX,custom_loss)
opt = keras.optimizers.Adam(learning_rate=current_learning_rate)
autoencoder.compile(optimizer=opt)

(x,a),y= next(loader_train)
print(y)
_ = autoencoder.train_step([(tf.convert_to_tensor(x)),
                                      tf.convert_to_tensor(a),
                                      y])
autoencoder.built = True

losses_all_train = []
losses_all_test = []

if os.path.exists(PATH_OUT+MODEL_NAME):
    print("Trying to load the model")
    print("PATH:",PATH_OUT+MODEL_NAME)
    losses_all_train, losses_all_test = load_model(PATH_OUT,MODEL_NAME,autoencoder)

else:
    os.makedirs(PATH_OUT+MODEL_NAME)


path_args = PATH_OUT+MODEL_NAME+"/args.txt"
save_config(parsed_args,path_args)

epochs = parsed_args.epochs
steps_train = math.ceil(train.n_graphs/parsed_args.batchsize)
steps_test = math.ceil(test.n_graphs/parsed_args.batchsize)

for e in range(len(losses_all_train),epochs):

    if current_learning_rate > parsed_args.learningrate * pow(0.7,floor(e/40)):
        current_learning_rate = parsed_args.learningrate * pow(0.7,floor(e/40))
        opt.lr.assign(current_learning_rate)
    print("EPOCH",e)
    loss=None
    losses_all = []
    for _ in range(steps_train):        
        (x,a),y= next(loader_train)
        loss = autoencoder.train_step([tf.convert_to_tensor(x),
                                      tf.convert_to_tensor(a),
                                      y])
        losses_all.append([float(keras.backend.get_value(loss[l])) for l in loss])
        if tf.math.is_nan(loss['loss']):
            print("LOSS == NAN")
            break
        
    if tf.math.is_nan(loss['loss']):
        print("LOSS == NAN")
        break
    avg_losses_all = np.mean(losses_all,axis=0)
    np.set_printoptions(suppress=True,precision=3)
    print(avg_losses_all)
    autoencoder.set_weights_for_loss(avg_losses_all,e)
    losses_all_train.append([float(keras.backend.get_value(loss[l])) for l in loss])
    
    test_loses = test_model(autoencoder,loader_test,steps_test,variational)
    losses_all_test.append(test_loses)
    if e%5 == 0:
        save_model(PATH_OUT,MODEL_NAME,losses_all_train,losses_all_test,autoencoder,Variational=variational)
    print("Loss train: ",np.mean(avg_losses_all[0]))
    print("Loss test: ", np.mean(test_loses[0]))

print("EPOCH",epochs)
print("Loss train: ",np.mean(losses_all_train[-1][0]))
print("Loss test: ", np.mean(losses_all_test[-1][0]))

save_model(PATH_OUT,MODEL_NAME,losses_all_train,losses_all_test,autoencoder,Variational=variational)


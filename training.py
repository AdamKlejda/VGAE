import spektral.data as data
import tensorflow as tf
import numpy as np 
from tensorflow import keras
import os
from GraphDataset import GraphDataset
from autoencoder import Encoder, DecoderX, DecoderA, VGAE
import matplotlib.pyplot as plt
from custom_layers import *
import sys
import argparse
import math
import json

def ensureDir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def save_model(path,name,loss_all,autoencoder):
    loss_all = [keras.backend.get_value(x['loss']) for x in  losses_all]
    reconstruction_loss = [keras.backend.get_value(x['reconstruction_loss']) for x in  losses_all]
    reconstruction_lossA = [keras.backend.get_value(x['reconstruction_lossA']) for x in  losses_all]
    reconstruction_lossX = [keras.backend.get_value(x['reconstruction_lossX']) for x in  losses_all]
    kl_loss = [keras.backend.get_value(x['kl_loss']) for x in  losses_all]
    dic_losses = {"loss_all":str(loss_all),
                  "reconstruction_loss": str(reconstruction_loss),
                  "reconstruction_lossA":str(reconstruction_lossA),
                  "reconstruction_lossX":str(reconstruction_lossX),
                  "kl_loss":str(kl_loss)}
         
    with open("{0}{1}/losses.json".format(path,name), "w") as outfile: 
        json.dump(dic_losses, outfile)

    fig, ax = plt.subplots(2,3,figsize=(10,15))
    x = np.array(loss_all)
    ax[0,0].plot(x)
    ax[0,0].set(xlabel='epoch', ylabel='loss')

    x = np.array(kl_loss)
    ax[0,1].plot(x)
    ax[0,1].set(xlabel='epoch', ylabel='kl_loss')

    x = np.array(reconstruction_loss)
    ax[1,0].plot(x)
    ax[1,0].set(xlabel='epoch', ylabel='reconstruction_loss')

    x = np.array(reconstruction_lossX)
    ax[1,1].plot(x)
    ax[1,1].set(xlabel='epoch', ylabel='reconstruction_lossX')

    x = np.array(reconstruction_lossA)
    ax[1,2].plot(x)
    ax[1,2].set(xlabel='epoch', ylabel='reconstruction_lossA')

    fig.savefig("{0}{1}/loss.png".format(path,name))
    autoencoder.save_weights("{0}{1}/model".format(path,name))
    pass

def parseArguments():
    parser = argparse.ArgumentParser(
        description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[
            0])
    parser.add_argument('-pathframs', type=ensureDir, required=True, help='Path to the Framsticks library without trailing slash.')
    parser.add_argument('-pathdata', type=ensureDir, required=True, help='Path to the data input in gen format.')
    parser.add_argument('-pathout', type=ensureDir, required=True, help='Path to save output model and plots.')
    parser.add_argument('-batchsize', type=int, required=False, help='',default=256)
    parser.add_argument('-adjsize', type=int, required=False, help='',default=15)
    parser.add_argument('-numfeatures', type=int, required=False, help='',default=3)
    parser.add_argument('-latentdim', type=int, required=False, help='',default=10)
    parser.add_argument('-nhidden', type=int,   required=False, help='',default=64)
    parser.add_argument('-convenc', type=int, required=False, help='',default=0)
    parser.add_argument('-denseenc', type=int, required=False, help='',default=4)
    parser.add_argument('-densedeca', type=int, required=False, help='',default=3)
    parser.add_argument('-convdecx', type=int, required=False, help='',default=1)
    parser.add_argument('-densedecx', type=int, required=False, help='',default=3)
    parser.add_argument('-learningrate', type=float, required=False, help='',default=0.0001)
    parser.add_argument('-epochs', type=int, required=False, help='',default=500)

    parser.set_defaults(debug=False)
    return parser.parse_args()

parsed_args = parseArguments()

PATH_OUT = (str(parsed_args.pathout)+
            "numfeatures"+str(parsed_args.numfeatures) +
            "/adjsize"+str(parsed_args.adjsize) + 
            "/batchsize"+str(parsed_args.batchsize) +
            "/latentdim"+str(parsed_args.latentdim)+
            "/nhidden"+str(parsed_args.nhidden)+
            "/learningrate"+str(parsed_args.learningrate)+
            "/"
            )
MODEL_NAME = ("model_enc_"+str(parsed_args.convenc)+"_"+str(parsed_args.denseenc)+
             "_deca"+str(parsed_args.densedeca)+
             "_decx"+str(parsed_args.convdecx)+"_"+str(parsed_args.densedecx))

if not os.path.exists(PATH_OUT+MODEL_NAME):
    os.makedirs(PATH_OUT+MODEL_NAME)

with open(PATH_OUT+MODEL_NAME+"/args.txt", "w") as f:
    json.dump(parsed_args.__dict__, f, indent=2)

myData = GraphDataset(parsed_args.pathframs, parsed_args.pathdata,size_of_adj=parsed_args.adjsize)
# dataset.apply(normalize_one())
loader = data.BatchLoader(myData, batch_size=parsed_args.batchsize)

encoder = Encoder(latent_dim=parsed_args.latentdim,
                  n_hidden=parsed_args.nhidden,
                  num_conv=parsed_args.convenc,
                  num_dense=parsed_args.denseenc)
decoderA = DecoderA(adjency_size=parsed_args.adjsize,
                    latent_dim=parsed_args.latentdim,
                    num_dense=parsed_args.densedeca)
decoderX = DecoderX(latent_dim=parsed_args.latentdim,
                    adjency_size=parsed_args.adjsize,
                    num_features=parsed_args.numfeatures,
                    num_conv=parsed_args.convdecx,
                    num_dense=parsed_args.densedecx)


inputsX = tf.keras.Input(shape=(parsed_args.adjsize,parsed_args.numfeatures))
inputsA = tf.keras.Input(shape=(parsed_args.adjsize,parsed_args.adjsize))
inp= [inputsX,inputsA]

autoencoder = VGAE(encoder,decoderA,decoderX)

opt = keras.optimizers.Adam(learning_rate=parsed_args.learningrate)
autoencoder.compile(optimizer=opt)
autoencoder._set_inputs(inp)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

losses_all = []

epochs = parsed_args.epochs
# steps = math.ceil(myData.n_graphs/parsed_args.batchsize)
steps=10
for e in range(epochs):
    print("EPOCH",e)
    loss=None
    avg_loss = []
    for i in range(steps):        
        x,a = next(loader)
        loss = autoencoder.train_step([(tf.convert_to_tensor(x)),
                                      tf.convert_to_tensor(a)])
        avg_loss.append(loss['loss'])
        if tf.math.is_nan(loss['loss']):
            print("LOSS == NAN")
            break
    if tf.math.is_nan(loss['loss']):
        print("LOSS == NAN")
        break
    losses_all.append(loss)
    if e%20 == 0:
        save_model(PATH_OUT,MODEL_NAME,losses_all,autoencoder)
    print("LOSS: ",np.mean(avg_loss))

save_model(PATH_OUT,MODEL_NAME,losses_all,autoencoder)


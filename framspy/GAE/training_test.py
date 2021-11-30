import spektral.data as data
import tensorflow as tf
import numpy as np 
from tensorflow import keras
import os
import sys
import argparse
import math
from math import floor
import pandas as pd 
import pickle

from GAE.GraphDataset import GraphDataset
from GAE.autoencoder import EncoderGAE,EncoderVGAE, DecoderX, DecoderA, VGAE, GAE
from GAE.utils import *
from GAE.custom_layers import *
from GAE.LossManager import LossManager, LossTypes
from GAE.framasToGraph import FramsTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


NUM_EXAMP=1000

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
        x,a = next(data_loader)
        if variational:
            total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX,kl_loss = autoencoder.test_step([(tf.convert_to_tensor(x)),
                                    tf.convert_to_tensor(a)])
            loss_all.append([total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX,kl_loss])

        else:
            total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX = autoencoder.test_step([(tf.convert_to_tensor(x)),
                                    tf.convert_to_tensor(a)])
            loss_all.append([total_loss,reconstruction_loss,reconstruction_lossA,reconstruction_lossX])
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
    parser.add_argument('-numfeatures', type=int, required=False, help='',default=3)
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

train, test = GraphDataset(parsed_args.pathframs, parsed_args.pathdata,fitness=fitness,size_of_adj=parsed_args.adjsize).read()


# frams_transf = FramsTransformer(parsed_args.pathframs,parsed_args.adjsize)
# best_gen = '//0\np:2.298312727204787, 2.298312727204787, 2.298312727204787\np:1.0942442864477269, 3.8224330349472115, 2.298312727204787\np:0.7028104614021384, 4.270621270802676, 4.175778174543551\np:1.565672987248388, 3.759406037940308, 5.8569109926396425\np:1.4783265317666054, 4.289475890299355, 7.75219449408074\np:0.7772495399273303, 4.088782239960528, 9.58258642105356\np:4.29525112746713, 2.403939894437829, 2.298312727204787\np:1.391318972254275, 0.5160999449083605, 2.298312727204787\np:1.1953890754817635, 0.0, 4.190603268140904\np:1.3044577547232905, 0.0002276606071149878, 6.157922176447118\np:1.614924086010032, 0.32638985797429987, 8.017652350167477\np:1.7004047886718914, 0.7839183371053251, 9.930519748252479\nj:0, 1\nj:0, 6\nj:0, 7\nj:1, 2\nj:2, 3\nj:3, 4\nj:4, 5\nj:7, 8\nj:8, 9\nj:9, 10\nj:10, 11\n'
# best_gen = frams_transf.getGrafFromString(best_gen)

loader_train = data.BatchLoader(train, batch_size=NUM_EXAMP)
loader_start = data.BatchLoader(test, batch_size=parsed_args.batchsize)

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

(x,a),y = next(loader_start)
_ = autoencoder.train_step([(tf.convert_to_tensor(x)),
                                      tf.convert_to_tensor(a),
                                      y])
autoencoder.built = True

losses_all_train = []
losses_all_test = []

if os.path.exists(PATH_OUT+MODEL_NAME):
    print("Trying to load the model")
    losses_all_train, losses_all_test = load_model(PATH_OUT,MODEL_NAME,autoencoder)

else:
    os.makedirs(PATH_OUT+MODEL_NAME)


path_save = PATH_OUT+MODEL_NAME
print(path_save)
epochs = parsed_args.epochs

ranges = [0.0001,0.000333,0.001,0.00333,0.01,0.0333,0.1,0.333,1,3.333]

(x,a),y = next(loader_train)

if variational == True:
    z_mean, z_log_var, z_training  = autoencoder.encoder([tf.convert_to_tensor(x),
                             tf.convert_to_tensor(a)])
else:
    z_training = autoencoder.encoder([tf.convert_to_tensor(x),
                             tf.convert_to_tensor(a)])
z_random = np.random.uniform(-10,10,size = (NUM_EXAMP,parsed_args.latentdim))

def generate_shifted(z, ranges,name):
    for r in ranges:
        shift = np.random.uniform(-r,r,size = (NUM_EXAMP,parsed_args.latentdim))
        shifted_z = z+shift
        dec_a = autoencoder.decoderA(tf.convert_to_tensor(z))
        dec_x = autoencoder.decoderX([tf.convert_to_tensor(z),dec_a])
        
        shifted_dec_a = autoencoder.decoderA(tf.convert_to_tensor(shifted_z))
        shifted_dec_x = autoencoder.decoderX([tf.convert_to_tensor(shifted_z),shifted_dec_a])
        df_all = pd.DataFrame()
        for i in range(NUM_EXAMP):
            data_temp = {

                'decA' :[dec_a[i].numpy()],
                'decX' :[dec_x[i].numpy()],
                'z':[z[i]],
                'shifted_dec_a' :[shifted_dec_a[i].numpy()],
                'shifted_dec_x' :[shifted_dec_x[i].numpy()],
                'shifted_z' :[shifted_z[i]],
            }

            data_df = pd.DataFrame(data_temp)
            df_all = df_all.append(data_df)

        df_all.to_pickle(path_save+"/{0}_{1}_{2}.pkl".format(name,-r,r))

generate_shifted(z_training,ranges,"z_training")
print("generated z_training")
generate_shifted(z_random,ranges,"z_random")
print("generated z_random")

# num_of_examples = 10
# steps = 11
# ranges = [[0,0.00001],[0,0.0001],[0,0.001],[0,0.01],[0,1],[0,1],[0,10],[0,100],[0,1000]]

# for r in ranges:
#     z_rand = np.random.uniform(r[0],r[1],size = (num_of_examples,parsed_args.latentdim))
#     # print(type(z),z.shape)
#     df_all = []
#     for z in z_rand:
#         finall = np.zeros(shape=(steps,len(z)))

#         for i in range(len(z)):
#             new = np.linspace(z[i],-z[i],steps)
#             for j in range(len(new)):
#                 finall[j][i] = new[j]
#         # print(type(finall),finall.shape)
        
#         df_z = pd.DataFrame()
#         for i in range(len(finall)):
#             dec_a = autoencoder.decoderA(tf.convert_to_tensor([finall[i]]))
#             dec_x = autoencoder.decoderX([tf.convert_to_tensor([finall[i]]),dec_a])
#             # print(dec_a)
#             # print(dec_a[0])
#             data_temp = {
#                 'decA' :[dec_a[0].numpy()],
#                 'decX' :[dec_x[0].numpy()],
#                 'z':[finall[i]],
#             }
#             # print()

#             data_df = pd.DataFrame(data_temp)
#             df_z = df_z.append(data_df)
#         df_all.append(df_z)
#     # df_all.to_pickle(path_save+"/reflection_{0}_{1}.pkl".format(r[0],r[1]))
#     # print(df_all)
#     with open(path_save+"/reflection_{0}_{1}.pkl".format(r[0],r[1]), 'wb') as fh:
#         pickle.dump(df_all, fh)


# epochs = parsed_args.epochs
# steps_train = math.ceil(train.n_graphs/parsed_args.batchsize)
# steps_test = math.ceil(test.n_graphs/parsed_args.batchsize)

# num_of_examples = 200

# x,a = next(loader_train)           
# dec_x, dec_a= autoencoder.call([tf.convert_to_tensor(x),tf.convert_to_tensor(a)])

# df_all = pd.DataFrame()
# for i in range(num_of_examples):
#     data_temp = {
#         'trueA':[a[i]],
#         'trueX':[x[i]],
#         'decA' :[dec_a[i].numpy()],
#         'decX' :[dec_x[i].numpy()],
#     }
#     data_df = pd.DataFrame(data_temp)
#     df_all = df_all.append(data_df)
#     np.set_printoptions(precision=0,suppress=True)
#     print("TRUE A")
#     print(a[i])
#     print("RECONSTRUCTED A")
#     print(dec_a[i])
#     np.set_printoptions(precision=2,suppress=True)

#     print("TRUE X")
#     print(x[i])
#     print("RECONSTRUCTED X")
#     print(dec_x[i])
#     print("\n\n\n\n")

import spektral.data as data
import tensorflow as tf
import numpy as np 
from tensorflow import keras
import os
import sys
import argparse
# from math import floor

from GAE.GraphDataset import GraphDataset
from GAE.autoencoder import EncoderGAE,EncoderVGAE, DecoderX, DecoderA, VGAE, GAE
from GAE.utils import *
from GAE.custom_layers import *
from GAE.custom_layers import ConvTypes
from GAE.LossManager import LossManager, LossTypes
from GAE.framasToGraph import FramsTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def ensureDir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
        
def get_convType(name):
    if name =="gcnconv":
        return ConvTypes.GCNConv
    elif name =="armaconv":
        return ConvTypes.ARMAConv
    elif name =="eccconv":
        return ConvTypes.ECCConv
    elif name =="gatconv":
        return ConvTypes.GATConv
    elif name =="gcsconv":
        return ConvTypes.GCSConv
    
def get_Loss(name):
    if name =="joints":
        return LossTypes.joints
    elif name =="parts":
        return LossTypes.parts
    elif name =="fitness":
        return LossTypes.fitness
    elif name =="dissim":
        return LossTypes.dissim
    elif name =="None":
        return LossTypes.No
    elif name == "No":
        return LossTypes.No


def load_config(pathconfig):
    with open(pathconfig) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    params_dict = {} 
    params_dict['pathframs']="/home/inf131778/Framsticks50rc19/"
    params_dict['pathdata']=lines[1]
    params_dict['pathout']="/home/inf131778/VGAE/framspy/models_vertpos/"
    params_dict['batchsize']=int(lines[3])
    params_dict['adjsize']=int(lines[4])
    params_dict['numfeatures']=int(lines[5])
    params_dict['latentdim']=int(lines[6])
    params_dict['nhidden']=int(lines[7])
    params_dict['convenc']=int(lines[8])
    params_dict['denseenc']=int(lines[9])
    params_dict['densedeca']=int(lines[10])
    params_dict['convdecx']=int(lines[11])
    params_dict['densedecx']=int(lines[12])
    params_dict['learningrate']=float(lines[13])
    params_dict['epochs']=int(lines[14])
    params_dict['convtype']=get_convType(lines[15])
    params_dict['variational']=lines[16]
    params_dict['loss']=get_Loss(lines[17])
    params_dict['trainid']=lines[18]
    return params_dict

             
             
class AE_evolalg:
    def __init__(self,path_config) -> None:
        self.params_dict = load_config(path_config)

        self.frams_manager = FramsManager(self.params_dict['pathframs'])
        self.frams_transformer = FramsTransformer(self.params_dict['pathframs'],self.params_dict['adjsize'])
        ae_type=None
        if self.params_dict['variational'] == "True":
            self.variational=True
        else:
            self.variational=False

        if self.variational == True:
            ae_type="VGAE"
        else:
            ae_type="GAE"

        self.path_out = (str(self.params_dict['pathout'])+
                    str(self.params_dict['loss'])+
                    "/"+ae_type +
                    "/numfeatures"+str(self.params_dict['numfeatures']) +
                    "/adjsize"+str(self.params_dict['adjsize']) + 
                    "/batchsize"+str(self.params_dict['batchsize']) +
                    "/latentdim"+str(self.params_dict['latentdim'])+
                    "/nhidden"+str(self.params_dict['nhidden'])+
                    "/learningrate"+str(self.params_dict['learningrate'])+
                    "/convtype"+str(self.params_dict['convtype'])+
                    "/"
                    )
        self.model_name = ("model_enc_"+str(self.params_dict['convenc'])+"_"+str(self.params_dict['denseenc'])+
                    "_deca"+str(self.params_dict['densedeca'])+
                    "_decx"+str(self.params_dict['convdecx'])+"_"+str(self.params_dict['densedecx'])+
                    "_train_id_"+str(self.params_dict['trainid'])
                    )
                    
        if self.params_dict['loss'] is not LossTypes.No:
            lossManager = LossManager(self.params_dict['pathframs'],"eval-allcriteria_new.sim","vertpos")
            if self.params_dict['loss'] == LossTypes.joints:
                custom_loss = lossManager.joints_too_big_loss
            elif self.params_dict['loss'] == LossTypes.parts:
                custom_loss = lossManager.part_number_loss
            elif self.params_dict['loss'] == LossTypes.fitness:
                custom_loss = lossManager.fitness_comparison_loss
            elif self.params_dict['loss'] == LossTypes.dissim:
                custom_loss = lossManager.dissimilarity_comparison
            else:
                print(self.params_dict['loss']," is not supported, custom loss set to None")
                custom_loss = None
        else:
            custom_loss = None

        train, test = GraphDataset(self.params_dict['pathframs'], self.params_dict['pathdata'],size_of_adj=self.params_dict['adjsize'],max_examples=500).read()

        loader_train = data.BatchLoader(train, batch_size=self.params_dict['batchsize'])
        loader_test = data.BatchLoader(test, batch_size=self.params_dict['batchsize'])

        if self.variational == True:
            encoder = EncoderVGAE(latent_dim=self.params_dict['latentdim'],
                            n_hidden=self.params_dict['nhidden'],
                            num_conv=self.params_dict['convenc'],
                            num_dense=self.params_dict['denseenc'],
                            convtype=self.params_dict['convtype'])
        else:
            encoder = EncoderGAE(latent_dim=self.params_dict['latentdim'],
                        n_hidden=self.params_dict['nhidden'],
                        num_conv=self.params_dict['convenc'],
                        num_dense=self.params_dict['denseenc'],
                        convtype=self.params_dict['convtype'])
                            
        decoderA = DecoderA(adjency_size=self.params_dict['adjsize'],
                            latent_dim=self.params_dict['latentdim'],
                            num_dense=self.params_dict['densedeca'])
        decoderX = DecoderX(latent_dim=self.params_dict['latentdim'], 
                            adjency_size=self.params_dict['adjsize'],
                            num_features=self.params_dict['numfeatures'],
                            num_conv=self.params_dict['convdecx'],
                            num_dense=self.params_dict['densedecx'],
                            convtype=self.params_dict['convtype'])


        current_learning_rate = self.params_dict['learningrate']

        if self.variational == True:
            self.autoencoder = VGAE(encoder,decoderA,decoderX,custom_loss)
        else:
            self.autoencoder = GAE(encoder,decoderA,decoderX,custom_loss)

        opt = keras.optimizers.Adam(learning_rate=current_learning_rate)
        self.autoencoder.compile(optimizer=opt)

        (x,a),y = next(loader_train)
        _ = self.autoencoder.train_step([(tf.convert_to_tensor(x)),
                                            tf.convert_to_tensor(a),
                                            y])
        self.autoencoder.built = True

        if os.path.exists(self.path_out+self.model_name+"/"):
            print("Trying to load the model")
            _, _  = load_model(self.path_out,self.model_name,self.autoencoder)

        else:
            os.makedirs(self.path_out+self.model_name)

        self.path_save = self.path_out+self.model_name

    def prepareGenList(self,X,A,idx):
        gen_list = gen_f0_from_tensors(X,A)
        gen_correct = []
        idx_correct = []
        ec = self.frams_manager.frams.MessageCatcher.new()
        for g in range(len(gen_list)):
            gen = self.frams_manager.check_consistency_for_gen(gen_list[g])
            if gen is not None:
                gen = self.frams_manager.reduce_joint_length_for_gen(gen)
            if gen is not None:
                gen_correct.append(gen)
                idx_correct.append(idx[g])
        ec.close()
        return idx_correct,gen_correct

    def prepareXAforPopulation(self,population):
        x_all = []
        a_all = []
        for p in population:
            graph = self.frams_transformer.getGrafFromString(p) 
            x_all.append(graph.x)
            a_all.append(graph.a)
        return x_all,a_all

    def mutate_population(self,population,idx,m_range=[-0.33,0.33]):
        # translate to a and x
        print("new genotype")
        print(len(population))
        x,a = self.prepareXAforPopulation(population)
        if self.variational== True:
            z_mean, z_log_var, orginal_z  = self.autoencoder.encoder([(tf.convert_to_tensor(x)),
                                        tf.convert_to_tensor(a)])
        else:
            orginal_z  = self.autoencoder.encoder([(tf.convert_to_tensor(x)),
                                        tf.convert_to_tensor(a)])
        # mutate by adding random stuff to z 
        new_z_list = []
        for z in orginal_z.numpy():
            mutation  = np.random.uniform(m_range[0],m_range[1],self.params_dict['latentdim'])
            m_z = z + mutation
            new_z_list.append(m_z)
        new_z = tf.convert_to_tensor(new_z_list)
        recA = self.autoencoder.decoderA(new_z)
        recX = self.autoencoder.decoderX([new_z,recA])

        idx, geno = self.prepareGenList(recX,recA,idx)
        print(len(geno),len(idx))
        return idx, geno




def parseArguments():
    parser = argparse.ArgumentParser(
        description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[
            0])
    parser.add_argument('-pathconfig', type=str, required=True, help='Path to the Framsticks library without trailing slash.')
    return parser.parse_args()

if __name__ == "__main__":
    parsed_args = parseArguments()
    autoencoder = AE_evolalg(parsed_args.pathconfig)
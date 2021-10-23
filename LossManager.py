from re import A, S

from tensorflow.python.ops.gen_control_flow_ops import no_op
from utils import gen_f0_from_tensors,FramsManager
from framspy.FramsticksLib import *
from scipy.spatial import distance
from enum import Enum
import numpy as np
import pandas as pd 
class LossTypes(Enum):
    joints = "joints"
    parts = "parts"
    fitness = "fitness"
    dissim = "dissim"
    No = "None"
    def __str__(self):
        return self.name


class LossManager:
    def __init__(self,pathFrams,path_to_sim="eval-allcriteria.sim",fitness="vertpos") -> None:
        self.FramsManager = FramsManager(pathFrams)
        self.framsLib = FramsticksLib(pathFrams,None,[path_to_sim])
        self.fitness = fitness

    def get_latent_dist(self,latent):
        dist_list = np.zeros(shape=(len(latent),len(latent)))
        for i in range(len(latent)-1):
            for j in range(i+1,len(latent)):
                dist_list[i,j]=distance.euclidean(latent[i],latent[j])
                dist_list[j,i]=distance.euclidean(latent[i],latent[j])
        return np.mean(dist_list, axis=1)

    def joints_too_big_loss(self,xa_orginal,xa_reconstructed,latent_space):
        x,a = xa_reconstructed
        gen_list = gen_f0_from_tensors(x,a)
        c_wrong_joints = self.FramsManager.count_wrong_joints(gen_list)
        return c_wrong_joints * 100

    def part_number_loss(self,xa_orginal,xa_reconstructed,latent_space):
        # frams with similar number of parts should be close to each other
        x,a= xa_orginal
        gen_list = gen_f0_from_tensors(x,a)
        n_parts_list = []
        for gen in gen_list:            
            m = self.FramsManager.frams.Model.newFromString(gen);
            n_parts = m.numparts._value()
            n_parts_list.append(n_parts)

        dist_list = self.get_latent_dist(latent_space)
        zipped=  zip(n_parts_list,dist_list)
        df = pd.DataFrame(zipped,columns=[0,1])
        my_r = df.corr(method="spearman")
        rho = my_r[0][1]
        return 1000*(1-rho)

    def fitness_comparison_loss(self,xa_orginal,xa_reconstructed,latent_space):
        # frams with similar fitness should be close to each other
        x,a= xa_orginal
        gen_list = gen_f0_from_tensors(x,a)
        c = self.framsLib.evaluate(gen_list)
        fit_list = [f['evaluations'][''][self.fitness] for f in c]
        dist_list = self.get_latent_dist(latent_space)
        zipped=  zip(fit_list,dist_list)
        df = pd.DataFrame(zipped,columns=[0,1])
        my_r = df.corr(method="spearman")
        rho = my_r[0][1]
        return 1000*(1-rho)

    def dissimilarity_comparison(self,xa_orginal,xa_reconstructed,latent_space):
        # similar frams should be close to each other
        x,a= xa_orginal
        gen_list = gen_f0_from_tensors(x,a)
        dissim_list = self.framsLib.dissimilarity(gen_list)
        dissim_list = np.mean(np.array(dissim_list), axis=1)
        dist_list = self.get_latent_dist(latent_space)
        zipped=  zip(dissim_list,dist_list)
        df = pd.DataFrame(zipped,columns=[0,1])
        my_r = df.corr(method="spearman")
        rho = my_r[0][1]
        return 1000*(1-rho)
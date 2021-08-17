from framsfiles import reader as framsreader
from framasToGraph import FramsTransformer
from spektral.data import Dataset
import fnmatch
import os
import numpy as np
import random 
import spektral

class GraphDataset(Dataset):

    transformer = None

    def __init__(self, path_frams, path_data ,n_samples=1000,size_of_adj=30,number_of_rep=100, **kwargs):
        self.n_samples = n_samples
        self.transformer = FramsTransformer(path_frams,size_of_adj)
        self.path_data = path_data
        self.size_of_adj = size_of_adj
        self.number_of_rep=number_of_rep
        
        super().__init__(**kwargs)

    def read(self):
        out = []
        dict_a_c = {}
        dict_a_x = {}
        counter=0
        for path, subdirs, files in os.walk(self.path_data):
            name = "*.gen"
            for file in fnmatch.filter(files, name):
                try:
                    halloffame_gen = framsreader.load(self.path_data+file, "gen file")
                    for g in halloffame_gen:
                        t = self.transformer.getGrafFromString(g['genotype'])
                        a_s = t.a.tostring()
#                         print(str(t.a))
                        if a_s in dict_a_c:
                            dict_a_c[a_s]+=1
                            if dict_a_c[a_s] <self.number_of_rep:
                                out.append(t)
                                dict_a_x[a_s].append(t.x)
                        else:
                            dict_a_c[a_s]=1
                            out.append(t)
                            dict_a_x[a_s]=[t.x]
                            
                    
                        
                except Exception as e:
                    print(e)
                    print("pass")
        for k in dict_a_c:
            c = self.number_of_rep - dict_a_c[k]
            if c>0:
                for i in range(c,1,-1):
                    a = np.fromstring(k).reshape([self.size_of_adj,self.size_of_adj])
                    l = len(dict_a_x[k])
                    res = random.randint(0, l-1)
                    x = dict_a_x[k][res]
                    x = np.where(x == -1, x, x+(c-i))
                    out.append(spektral.data.graph.Graph(x=x, a=a, e=None, y=None))
        return np.array(out)

from framsfiles import reader as framsreader
from framasToGraph import FramsTransformer
from spektral.data import Dataset
import fnmatch
import os


class GraphDataset(Dataset):

    transformer = None

    def __init__(self, path_frams, path_data ,n_samples=1000,size_of_adj=30, **kwargs):
        self.n_samples = n_samples
        self.transformer = FramsTransformer(path_frams,size_of_adj)
        self.path_data = path_data
        
        super().__init__(**kwargs)

    def read(self):
#         halloffame_gen = framsreader.load("/home/adam/thesis/testing_graphnn/halloffame.gen", "gen file")

#         # We must return a list of Graph objects
#         return [transformer.getGrafFromString(g['genotype']) for g in halloffame_gen]
        out = []
        counter=0
        for path, subdirs, files in os.walk(self.path_data):
        #     print(files)
        #     key = "{0}_{1}_{2}".format(fit[1],sim[1],gen)   
        #     print(key)

            name = "*.gen"
            for file in fnmatch.filter(files, name):
        #         print(file)
                try:
                    halloffame_gen = framsreader.load(self.path_data+file, "gen file")
                    for g in halloffame_gen:
                        out.append(self.transformer.getGrafFromString(g['genotype']))
                except Exception as e:
                    print(e)
#                 break
                counter +=1
#                 if counter > 10:
#                     break
        return out
#         break

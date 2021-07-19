import frams
import numpy as np
import spektral

path='/home/adam/Framsticks/Framsticks50rc19'

class FramsTransformer():
    frams = frams
    orto_max = 100
    
    def __init__(self,path,size_of_adj=10) -> None:
        self.frams.init(path)
        self.size_of_adj = size_of_adj

    def getAdjencyMatrix(self,model):
        aMatrix = np.zeros(shape=(model.numparts._value(),model.numparts._value()))
        aMatrix
#         print("NUM OF JOINGS:",model.numjoints._value())
        for i in range(model.numjoints._value()):
#             print("p1: ",model.getJoint(i).p1._value(), "p2: ",model.getJoint(i).p2._value() )
            aMatrix[model.getJoint(i).p1._value(),model.getJoint(i).p2._value()] = 1
        for i in range(aMatrix.shape[0]):
            aMatrix[i,i] = 1
        
#         HERE TO REMOVE IF DOESNT WORK
        pad = self.size_of_adj - model.numparts._value()
        aMatrix = np.pad(aMatrix,(0,pad),"constant",constant_values=(0))
        return aMatrix

    def getPartFeaturesArray(self,part):
        return np.array([part.x._value()/self.orto_max, #float
                        part.y._value() /self.orto_max,  #float
                        part.z._value() /self.orto_max,  #float
                        part.sh._value()/3, #0.1-3
                        part.s._value() /10,  #0.1-10
                        part.sx._value()/5, #0.05-5
                        part.sy._value()/5, #0.05-5
                        part.sz._value()/5, #0.05-5
                        part.rx._value(), #float
                        part.ry._value(), #float
                        part.rz._value(), #float 
                        part.dn._value()/5, #0.02-5
                        part.fr._value()/4, #0-4
                        part.ing._value(), #0-1
    #                      part.as._value()
    #                      sth for neurons...
                        ])
    def getFeaturesForParts(self,model):
        parts_features = []
        for i in range(model.numparts._value()):
            parts_features.append(self.getPartFeaturesArray(model.getPart(i)))
        
        pad = self.size_of_adj - model.numparts._value()
        features = 1
        if len(parts_features)>0:
            features=len(parts_features[0])
        parts_features = np.concatenate([parts_features,np.full((pad,features),-1)], axis=0)
        
        return np.array(parts_features)

    def getEdgeFeatures(self,joint):
        return np.array([joint.rx._value(),
                        joint.ry._value(),
                        joint.rz._value(),
                        joint.dz._value(),
                        joint.dx._value(),
                        joint.dy._value(),
                        joint.dz._value(),
                        joint.sh._value(),
                        joint.hx._value(),
                        joint.hy._value(),
                        joint.hz._value(),
                        joint.hrx._value(),
                        joint.hry._value(),
                        joint.hrz._value(),
                        joint.hxn._value(),
                        joint.hxp._value(),
                        joint.hyn._value(),
                        joint.hyp._value(),
                        joint.stif._value(),
                        joint.rotstif._value(),
                        joint.stam._value(),
    #                      neuron stuff
                        ])

    def getFeaturesForEdges(self,model): #JOINTS
        edges_features = []
        for i in range(model.numjoints._value()):
            edges_features.append(self.getEdgeFeatures(model.getJoint(i)))
        
        pad = self.size_of_adj - model.numjoints._value()
        features = 1
        if len(edges_features)>0:
            features=len(edges_features[0])
        edges_features = np.concatenate([edges_features,np.full((pad,features),-1)], axis=0)
        return np.array(edges_features)

    # def getNeuronsforModel(model):
    #     neuronsPartsFeatures = {}
    #     neuronsJointsFeatures = {}
    #     for i in range(model.numneurons._value()):
    #         model.getJoint(i)
    #     return 

    def graphFromModel(self,model):
    #     neuronsPartsFeatures, neuronsJointsFeatures = getNeuronsforModel(model)
        #FEATURES FOR NODE
        x = self.getFeaturesForParts(model)
    #     print(x)
        
        # Adjacency matrix
        a = self.getAdjencyMatrix(model)
    #     print("Adjacency matrix:\n", a) 
        
        
        # edge attributes
#         e = self.getFeaturesForEdges(model)
    #     print(e)
        # for the node or graph labels
        # y = 
        return spektral.data.graph.Graph(x=x, a=a, e=None, y=None)

    def getGrafFromString(self,str_model):
        m = self.frams.Model.newFromString(str_model)
        return self.graphFromModel(m)


geno1 = """//0
p:0.999, -0.428, -1.137, fr=0.205, ing=0.169, as=0.19
p:0.285, 0.532, -0.111, fr=0.328, ing=0.122, as=0.312
p:-0.24916222042317826, -0.026372082735363067, -0.4057733171945488, fr=0.222, ing=0.193, as=0.227
p:1.1463573899321884, -1.0892112107864644, -0.9040559228891072, fr=1.404, ing=0.623, as=0.364
p:0.808, -1.042, -0.316, as=0.151
p:0.835, -1.147, -0.372, fr=0.886, ing=0.623, as=0.364
p:0.818, -1.197, -0.21, fr=0.748, as=0.214
p:0.88, -1.204, -0.033, fr=0.748, as=0.214
p:0.973, -0.979, -0.297
j:0, 6, rotstif=0.875
j:8, 0, rotstif=0.875
j:1, 0, stif=0.975, rotstif=0.98
j:1, 8
j:4, 1, stif=0.803, rotstif=0.996, stam=0.22
j:1, 6, stif=0.798, rotstif=0.996, stam=0.22
j:2, 4, stif=0.998, stam=0.014
j:2, 6, stif=0.798, rotstif=0.996, stam=0.13
j:2, 1, stif=0.998, rotstif=0.892, stam=0.014
j:0, 4, rotstif=0.875
j:2, 0, stif=0.905, stam=0.309
j:3, 1
j:0, 3, rotstif=0.829
j:2, 3, rotstif=0.819
j:6, 4, rotstif=0.98, stam=0.502
j:3, 4, stam=0.275
j:0, 7, stif=0.798, rotstif=0.996, stam=0.13
j:8, 5
n:d=*
n:p=0, d=T
n:j=7, d=@
n:p=2, d=T
n:p=1, d=Gpart
n:d="N:in=0.895, fo=0.358, si=999.0, s=-0.43"
n:p=3, d=Gpart:rz=2.612
n:p=3, d=Gpart
n:p=5, d=Gpart:rz=2.612
n:p=5, d=Gpart
n:j=16, d=@
n:p=4, d=Gpart
c:10, 11
"""
# geno2 = "XXLXXXCXXC"

transformer = FramsTransformer(path)

m = transformer.getGrafFromString(geno1)
print(m)
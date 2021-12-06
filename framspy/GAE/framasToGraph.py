import frams
import numpy as np
import spektral

class FramsTransformer():
    frams = frams
    orto_max = 30
    
    def __init__(self,path,size_of_adj=30):
        self.frams.init(path)
        self.size_of_adj = size_of_adj

    def getAdjencyMatrix(self,model):
        aMatrix = np.zeros(shape=(model.numparts._value(),model.numparts._value()))
        aMatrix
#         print("NUM OF JOINGS:",model.numjoints._value())
        for i in range(model.numjoints._value()):
#             print("p1: ",model.getJoint(i).p1._value(), "p2: ",model.getJoint(i).p2._value() )
            aMatrix[model.getJoint(i).p1._value(),model.getJoint(i).p2._value()] = 1
            aMatrix[model.getJoint(i).p2._value(),model.getJoint(i).p1._value()] = 1
        for i in range(aMatrix.shape[0]):
            aMatrix[i,i] = 1
        
#         HERE TO REMOVE IF DOESNT WORK
        pad = self.size_of_adj - model.numparts._value()
        aMatrix = np.pad(aMatrix,(0,pad),"constant",constant_values=(0))
        return aMatrix

    def getPartFeaturesArray(self,part):
        x = 50
        return np.array([part.x._value(),#    #float
                        part.y._value() ,#    #float
                        part.z._value() ,#    #float
#                         part.sh._value()    #0.1-3
#                         part.s._value()     #0.1-10
#                         part.sx._value()    #0.05-5
#                         part.sy._value()    #0.05-5
#                         part.sz._value()    #0.05-5
#                         part.rx._value()    #float
#                         part.ry._value()    #float
#                         part.rz._value()    #float 
#                         part.dn._value()    #0.02-5
#                         part.fr._value()    #0-4
#                         part.ing._value()   #0-1
    #                      part.as._value()
    #                      sth for neurons...
                        1
                        ])
    def getFeaturesForParts(self,model):
        parts_features = []
        for i in range(model.numparts._value()):
            parts_features.append(self.getPartFeaturesArray(model.getPart(i)))
        
        pad = self.size_of_adj - model.numparts._value()
        features = 1
        if len(parts_features)>0:
            features=len(parts_features[0])
        parts_features = np.array(parts_features)
        n_base = len(parts_features)

        parts_features = np.concatenate([parts_features,np.full((pad,features),0)], axis=0)
        parts_features[n_base:,-1]=-1
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
        x = np.array(self.getFeaturesForParts(model))
    #     print(x)
        
        # Adjacency matrix
        a = np.array(self.getAdjencyMatrix(model))
    #     print("Adjacency matrix:\n", a) 
        
        
        # edge attributes
#         e = self.getFeaturesForEdges(model)
    #     print(e)
        # for the node or graph labels
        y = 0
        return spektral.data.graph.Graph(x=x, a=a, e=None, y=y)

    def getGrafFromString(self,str_model):
        m = self.frams.Model.newFromString(str_model)
        return self.graphFromModel(m)
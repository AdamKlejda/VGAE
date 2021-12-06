import numpy as np
import matplotlib.pyplot as plt
import frams
from scipy.spatial import distance

def save_model(path,name,losses_all_train,losses_all_test,autoencoder,Variational = False):

    test_loss_all = [x[0] for x in  losses_all_test]
    test_reconstruction_loss = [x[1] for x in  losses_all_test]
    test_reconstruction_lossA = [x[2] for x in  losses_all_test]
    test_reconstruction_lossX = [x[3] for x in  losses_all_test]
    if Variational:   
        test_kl_loss = [x[4] for x in  losses_all_test]

    train_loss_all = [x[0] for x in  losses_all_train]
    train_reconstruction_loss = [x[1] for x in  losses_all_train]
    train_reconstruction_lossA = [x[2] for x in  losses_all_train]
    train_reconstruction_lossX = [x[3] for x in  losses_all_train]
    if Variational:   
        train_kl_loss = [x[4] for x in  losses_all_train]

    with open("{0}{1}/losses_test.npy".format(path,name), "wb") as outfile: 
        np.save(outfile, np.array(losses_all_test))
    with open("{0}{1}/losses_train.npy".format(path,name), "wb") as outfile: 
        np.save(outfile, np.array(losses_all_train))

    # fig, ax = plt.subplots(2,3,figsize=(10,15))
    # x_train = np.array(train_loss_all)
    # x_test = np.array(test_loss_all)
    # ax[0,0].plot(x_train, label = "train")
    # ax[0,0].plot(x_test, label = "test")
    # ax[0,0].set(xlabel='epoch', ylabel='loss')

    # if Variational:
    #     x_train = np.array(train_kl_loss)
    #     x_test = np.array(test_kl_loss)
    #     ax[0,1].plot(x_train, label = "train")
    #     ax[0,1].plot(x_test, label = "test")
    #     ax[0,1].set(xlabel='epoch', ylabel='kl_loss')

    # x_train = np.array(train_reconstruction_loss)
    # x_test = np.array(test_reconstruction_loss)
    # ax[1,0].plot(x_train, label = "train")
    # ax[1,0].plot(x_test, label = "test")
    # ax[1,0].set(xlabel='epoch', ylabel='reconstruction_loss')

    # x_train = np.array(train_reconstruction_lossX)
    # x_test = np.array(test_reconstruction_lossX)
    # ax[1,1].plot(x_train, label = "train")
    # ax[1,1].plot(x_test, label = "test")
    # ax[1,1].set(xlabel='epoch', ylabel='reconstruction_lossX')

    # x_train = np.array(train_reconstruction_lossA)
    # x_test = np.array(test_reconstruction_lossA)
    # ax[1,2].plot(x_train, label = "train")
    # ax[1,2].plot(x_test, label = "test")
    # ax[1,2].set(xlabel='epoch', ylabel='reconstruction_lossA')

    # fig.savefig("{0}/loss_{1}.png".format(path,name))
    # fig.clear()
    # plt.close(fig)
    
    autoencoder.save_weights("{0}{1}/model.hdf5".format(path,name))
    print("Model saved")




def load_model(path,name,autoencoder):
    losses_all_test = []
    losses_all_train = []
    try:
        autoencoder.load_weights("{0}{1}/model.hdf5".format(path,name))
        print("Model loaded successfully")
    except Exception as e:
        print("Error while loading weights ", e)
    try:
        with open("{0}{1}/losses_test.npy".format(path,name), "rb") as infile: 
            losses_all_test = np.load(infile).tolist()
        with open("{0}{1}/losses_train.npy".format(path,name), "rb") as infile: 
            losses_all_train = np.load(infile).tolist()
        print("Losses loaded successfully")
    except Exception as e:
        print("Error while loading Losses ", e)
    return losses_all_train, losses_all_test


def roundXA(x,a):
    for (i1,i2),r in np.ndenumerate(a):
        if a[(i1,i2)] + 0.5 >= 1:
            a[(i1,i2)] = 1
        else:
            a[(i1,i2)] = 0
    # x= np.around(x,decimals=4)
    return x, a

# def repairA(x,a):
#     # make connection if only 1 has it
#     a_size = len(a[0])
#     for (i1,i2),r in np.ndenumerate(a):
#         if a[(i1,i2)]==1:
#             a[(i2,i1)] = 1
#     # remove connection to part that doesnt exist
#     num_parts = -1
#     for partId in range(len(x)):
#         if sum(x[partId]) < 0:
#             a[partId] = np.zeros(a_size)
#         else:
#             num_parts+=1
    
#     for i in range(num_parts):
#         if sum(a[i]) < 2:
#             if (i == a_size-2) or (i == a_size-1):
#                 a[(i,i-1)]=1
#                 a[(i-1,i)]=1
#             else:
#                 a[(i,i+1)]=1
#                 a[(i+1,i)]=1
#     return x, a
            
def add_part(g,p):
    return g + "p:"+str(p[0])+", "+str(p[1])+", "+str(p[2])+"\n"

def add_joint(g,i1,i2):
    return g + "j:"+str(i1)+", "+str(i2)+"\n"

def generateF1fromXA(x,a):
    genotype = "//0\n"
    counter_p = 0
    for part in x:
        if part[-1]>=0:
            genotype = add_part(genotype,part)
            counter_p+=1
    for (i1,i2),r in np.ndenumerate(a):
        if (a[(i1,i2)]==1 ) and (i1 != i2) and(i2>i1):
            if i1 <=counter_p and i2 <=counter_p:
                genotype = add_joint(genotype,i1,i2)
    genotype +=""
    return genotype


def gen_f0_from_df(df,max_elements=99999):
    frams_gen = []
    z_all = []
    for index, row in df.iterrows():
        decx = row[0] 
        deca = row[1]
        z = row[2]
        decx,deca = roundXA(decx,deca)
        f1 = generateF1fromXA(decx,deca)
        frams_gen.append(f1)
        z_all.append(z)
        if max_elements < len(frams_gen):
            return frams_gen,z_all
    return frams_gen,z_all

def gen_f0_from_tensors(decx,deca):
    frams_gen = []
    for i in range(len(decx)):

        x = decx[i].numpy()
        a = deca[i].numpy()

        x,a = roundXA(x,a)
        f1 = generateF1fromXA(x,a)
        frams_gen.append(f1)
    return frams_gen
    
def gen_f0_from_model(m):
    num_joints = m.numjoints._value() 
    num_parts = m.numparts._value()    
    genotype = "//0\n"    
    for p in range(num_parts):
        part = (m.getPart(int(p)).x._value(),m.getPart(int(p)).y._value(),m.getPart(int(p)).z._value())
        genotype = add_part(genotype,part)
    for j in range(num_joints):
        i1,i2 = m.getJoint(j).p1._value(),m.getJoint(j).p2._value()
        genotype = add_joint(genotype,i1,i2)
    return  genotype

class FramsManager():
    frams = frams

    def __init__(self,path) -> None:
        self.frams.init(path)

    def count_wrong_joints(self,gen_list):
        count =0
        ec = self.frams.MessageCatcher.new()
        for gen in gen_list:
            m = self.frams.Model.newFromString(gen)
            num_joints = m.numjoints._value() 
            for n in range(num_joints):            
                p,c = m.getJoint(n).p1._value(),m.getJoint(n).p2._value()
                p1 = (m.getPart(int(p)).x._value(),m.getPart(int(p)).y._value(),m.getPart(int(p)).z._value())
                p2 = (m.getPart(int(c)).x._value(),m.getPart(int(c)).y._value(),m.getPart(int(c)).z._value())
                dist =distance.euclidean(p1,p2)
                if dist > 2.0:
                    count +=1  
        ec.close()
        return count

    def merge_sub_groups(self,groups_inside_graph):
        end = False
        while end == False: 
            reset = False
            num_of_groups = len(groups_inside_graph)
            for i in range(num_of_groups-1):
                for j in range(i+1,num_of_groups):
                    if len(set(groups_inside_graph[i]).intersection(groups_inside_graph[j]))>0:
                        groups_inside_graph[i] = list(set(groups_inside_graph[i]+ groups_inside_graph[j]))
                        groups_inside_graph[j] = []
                        reset = True
                        break
                if reset == True:
                    break
            groups_inside_graph = [x for x in groups_inside_graph if x]
            if len(groups_inside_graph) == 1:
                end = True
            if reset == False:
                end = True
        return groups_inside_graph

    def create_groups(self,m):
        num_joints = m.numjoints._value() 
            
        groups_inside_graph=[]

        if num_joints == 0:
            num_parts = m.numparts._value() 
            for p in range(num_parts):
                groups_inside_graph.append([p])
            return groups_inside_graph
        for n in range(num_joints): 
            if len(groups_inside_graph) == 0:
                groups_inside_graph.append([m.getJoint(n).p1._value(),m.getJoint(n).p2._value()])
            else:
                in_g = False
                for g in groups_inside_graph:
                    if m.getJoint(n).p1._value() in g:
                        if m.getJoint(n).p2._value() not in g:
                            g.append(m.getJoint(n).p2._value())
                        in_g = True
                        break
                    if m.getJoint(n).p2._value() in g:
                        if m.getJoint(n).p1._value() not in g:
                            g.append(m.getJoint(n).p1._value())
                        in_g = True
                        break
                if in_g == False:
                    groups_inside_graph.append([m.getJoint(n).p1._value(),m.getJoint(n).p2._value()])

        return groups_inside_graph

    def add_joints_for_missing_parts(self,alone,all_connected_parts,m):
        joints=[]
        while len(alone)>0:
            connections = {}
            for p in alone:
                for c in all_connected_parts:                
                    p1 = (m.getPart(int(p)).x._value(),m.getPart(int(p)).y._value(),m.getPart(int(p)).z._value())
                    p2 = (m.getPart(int(c)).x._value(),m.getPart(int(c)).y._value(),m.getPart(int(c)).z._value())
                    dist =distance.euclidean(p1,p2)
                    connections[(p,c)]= dist
            joint = min(connections.keys(), key=connections.__getitem__)
            joints.append(joint)
            all_connected_parts.append(joint[0])
            alone.remove(joint[0])            
    #         print(joint,connections[joint])
        return joints

    def add_joints_to_connect_groups(self,groups_inside_graph,m):
        joints_to_add = []
        while len(groups_inside_graph) > 1:    
            connections = {}
            for i in range(len(groups_inside_graph)-1):
                for j in range(i+1,len(groups_inside_graph)):
                    for p in groups_inside_graph[i]:
                        for c in groups_inside_graph[j]:
                            p1 = (m.getPart(int(p)).x._value(),m.getPart(int(p)).y._value(),m.getPart(int(p)).z._value())
                            p2 = (m.getPart(int(c)).x._value(),m.getPart(int(c)).y._value(),m.getPart(int(c)).z._value())
                            dist =distance.euclidean(p1,p2)
                            connections[(p,c,i,j)]= dist
            joint = min(connections.keys(), key=connections.__getitem__)
            groups_inside_graph[joint[2]] = groups_inside_graph[joint[2]] + groups_inside_graph[joint[3]]
            groups_inside_graph[joint[3]] = []
            joints_to_add.append((joint[0],joint[1]))
            groups_inside_graph = [x for x in groups_inside_graph if x]
        return joints_to_add

    def check_consistency_for_gen(self, gen):
        try:
            m = self.frams.Model.newFromString(gen)
            n_parts = m.numparts._value()

            all_parts = np.arange(n_parts)
            groups_inside_graph = self.create_groups(m)
            all_connected_parts = []
            
            for g in groups_inside_graph:
                all_connected_parts = all_connected_parts + g

            alone = set(all_parts) - set(all_connected_parts)
            all_connected_parts = list(set(all_connected_parts))
            joints_to_add = self.add_joints_for_missing_parts(alone, all_connected_parts,m)

            for j in joints_to_add:
                gen =add_joint(gen,j[0],j[1])

            m = self.frams.Model.newFromString(gen)
            groups_inside_graph = self.create_groups(m)


            groups_inside_graph = self.merge_sub_groups(groups_inside_graph)
            joints_to_add = self.add_joints_to_connect_groups(groups_inside_graph,m)

            for j in joints_to_add:
                gen =add_joint(gen,j[0],j[1])

        except Exception as e:
            print(e)
            return None       

        return gen

    def reduce_joint_length_for_gen(self,gen):
        # print("GEN PASSED",[gen])
        m = frams.Model.newFromString(gen)
        num_joints = m.numjoints._value() 
        # print("num_joints start:",num_joints)
        num_of_iterations =0
        while True:
            joints_too_big=[]
            joints_too_small = []
            parts_connections = {}
            for n in range(num_joints):            
                p,c = m.getJoint(n).p1._value(),m.getJoint(n).p2._value()
                try:                
                    parts_connections[str(p)] +=1
                except:
                    parts_connections[str(p)] = 1
                try:               
                    parts_connections[str(c)] +=1
                except:
                    parts_connections[str(c)] = 1
                    
                p1 = (m.getPart(int(p)).x._value(),m.getPart(int(p)).y._value(),m.getPart(int(p)).z._value())
                p2 = (m.getPart(int(c)).x._value(),m.getPart(int(c)).y._value(),m.getPart(int(c)).z._value())
                dist =distance.euclidean(p1,p2)
                
                if dist >= 1.99999:
                    joints_too_big.append((dist,n,p,c))
                if dist == 0.0:
                    joints_too_small.append((dist,n,p,c))
                    
            if  len(joints_too_big) == 0 or num_of_iterations > 5 and len(joints_too_small) == 0:
                if len(joints_too_big) == 0 and len(joints_too_small) == 0:
                    # print("NJOINTS:",m.numjoints._value() )
                    return gen_f0_from_model(m)
                return None    

            for j in joints_too_big:
                reduce = j[2]
                leave = j[3]
                if parts_connections[str(j[2])] < parts_connections[str(j[3])]:
                    reduce = j[2]
                    leave = j[3]
                else:
                    reduce = j[3]
                    leave = j[2]
                    
                dist = j[0]
                end = 1.999999 
                what_we_want = dist/end     

                p1 = np.array([m.getPart(int(reduce)).x._value(),m.getPart(int(reduce)).y._value(),m.getPart(int(reduce)).z._value()])
                p2 = np.array([m.getPart(int(leave)).x._value(),m.getPart(int(leave)).y._value(),m.getPart(int(leave)).z._value()])

                p1_new  = (p2 - p1)/what_we_want
                p1_new = p1_new + p2

                m.getPart(int(reduce)).x,m.getPart(int(reduce)).y,m.getPart(int(reduce)).z = p1_new

            for j in joints_too_small:
                reduce = j[2]
                leave = j[3]
                if parts_connections[str(j[2])] < parts_connections[str(j[3])]:
                    reduce = j[2]
                    leave = j[3]
                else:
                    reduce = j[3]
                    leave = j[2]

                m.getPart(int(reduce)).x = m.getPart(int(reduce)).x._value() + 0.00001

            num_of_iterations +=1
        return None
        

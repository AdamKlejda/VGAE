import numpy as np
import matplotlib.pyplot as plt


def save_model(path,name,losses_all_train,losses_all_test,autoencoder,Variational = False):

    # test_loss_all = [x[0] for x in  losses_all_test]
    # test_reconstruction_loss = [x[1] for x in  losses_all_test]
    # test_reconstruction_lossA = [x[2] for x in  losses_all_test]
    # test_reconstruction_lossX = [x[3] for x in  losses_all_test]
    # test_reconstruction_lossMask = [x[4] for x in  losses_all_test]
    # if Variational:   
    #     test_kl_loss = [x[5] for x in  losses_all_test]

    # train_loss_all = [x[0] for x in  losses_all_train]
    # train_reconstruction_loss = [x[1] for x in  losses_all_train]
    # train_reconstruction_lossA = [x[2] for x in  losses_all_train]
    # train_reconstruction_lossX = [x[3] for x in  losses_all_train]
    # train_reconstruction_lossMask = [x[4] for x in  losses_all_train]
    # if Variational:   
    #     train_kl_loss = [x[5] for x in  losses_all_train]

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

    # x_train = np.array(train_reconstruction_lossMask)
    # x_test = np.array(test_reconstruction_lossMask)
    # ax[0,2].plot(x_train, label = "train")
    # ax[0,2].plot(x_test, label = "test")
    # ax[0,2].set(xlabel='epoch', ylabel='Mask')

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
    # plt.legend(loc="upper right")
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

def get_convType(name):
    from GAE.architecture.base.custom_layers import ConvTypes

    if name =="gcnconv":
        return ConvTypes.GCNConv
    elif name =="armaconv":
        return ConvTypes.ARMAConv
    elif name =="gatconv":
        return ConvTypes.GATConv
    elif name =="gcsconv":
        return ConvTypes.GCSConv
    
def get_Loss(name):
    from GAE.architecture.base.LossManager import LossTypes

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
    params_dict['pathframs']=lines[0]
    params_dict['pathdata']=lines[1]
    params_dict['pathout']=lines[2]
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
    # params_dict['trainid']=lines[18]
    return params_dict





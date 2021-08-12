PATH_FRAMS='/home/adam/Framsticks/Framsticks50rc19'
PATH_DATA ="graphs/gen/"
PATH_OUT = "models/"
EPOCHS = 2000
BATCH_SIZE = 258
ADJ_SIZE = 15
NUM_FEATURES = 3
LEARNING_RATE = 0.0001

# def create_filename(latentdim, nhidden, convenc, denseenc, densedeca, convdecx, densedecx):
#     return f"{latentdim}_{nhidden}_{convenc}_{denseenc}_{densedeca}_{convdecx}_{densedecx}"

def create_file(latentdim, nhidden, convenc, denseenc, densedeca, convdecx, densedecx):
    with open("configs/"+f"{latentdim}_{nhidden}_{convenc}_{denseenc}_{densedeca}_{convdecx}_{densedecx}", "w") as file:
        file.write(str(PATH_FRAMS)+"\n")    #pathframs
        file.write(str(PATH_DATA)+"\n")     #pathdata
        file.write(str(PATH_OUT)+"\n")      #pathout
        file.write(str(BATCH_SIZE)+"\n")    #batchsize
        file.write(str(ADJ_SIZE)+"\n")      #adjsize
        file.write(str(NUM_FEATURES)+"\n")  #numfeatures
        file.write(str(latentdim)+"\n")     #latentdim
        file.write(str(nhidden)+"\n")       #nhidden
        file.write(str(convenc)+"\n")       #convenc
        file.write(str(denseenc)+"\n")      #denseenc
        file.write(str(densedeca)+"\n")     #densedeca
        file.write(str(convdecx)+"\n")      #convdecx
        file.write(str(densedecx)+"\n")     #densedecx
        file.write(str(LEARNING_RATE)+"\n") #learningrate
        file.write(str(EPOCHS)+"\n")        #epochs

latentdim = [6,10]
nhidden = [64,128]
convenc = [1,3]
denseenc = [2,4]
densedeca = [1]
convdecx = [1]
densedecx = [2,5]

for l in latentdim:
    for nh in nhidden:
        for cv in convenc:
            for de in denseenc:
                for dda in densedeca:
                    for cdx in convdecx:
                        for ddx in densedecx:
                            create_file(l,nh,cv,de,dda,cdx,ddx)
PATH_FRAMS="~/"
PATH_DATA ="graphs/gen/"
PATH_OUT = "models/"
EPOCHS = 100
BATCH_SIZE = 256
ADJ_SIZE = 15
NUM_FEATURES = 3
LEARNING_RATE = 0.01

def create_file(variational,convtype,latentdim, nhidden, convenc, denseenc, densedeca, convdecx, densedecx, loss):
    with open("configs/"+f"v{variational}_{convtype}_{latentdim}_{nhidden}_{convenc}_{denseenc}_{densedeca}_{convdecx}_{densedecx}_{loss}", "w") as file:
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
        file.write(str(convtype)+"\n")      #convtype
        file.write(str(variational)+"\n")   #variational
        file.write(str(loss)+"\n")          #loss

latentdim = [15]
nhidden = [64]
convenc = [1,2]
denseenc = [2]
densedeca = [1]
convdecx = [1]
densedecx = [2]
convtypes = ["gcsconv","gcnconv","armaconv","gatconv",]
variational=['True','False']
loss= ['joints'] #['parts','fitness','dissim','None'] #
for v in variational:
    for l in latentdim:
        for nh in nhidden:
            for cv in convenc:
                for de in denseenc:
                    for dda in densedeca:
                        for cdx in convdecx:
                            for ddx in densedecx:
                                for ct in convtypes:
                                    for ls in loss:                                        
                                        create_file(v,ct,l,nh,cv,de,dda,cdx,ddx,ls)

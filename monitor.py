import os
import re
from pprint import pprint
from enum import Enum
import argparse
import sys 

def create_file_list():
    res = []
    for f in os.listdir("."):
        if "slurm" not in f:
            continue
        if ".out" not in f:
            continue
        res.append(f)
    return res

def get_file_name(file_name):
    name = ""
    r = "'pathframs':*"
    with open(file_name, "r") as f:
        for line in f:            
            a = re.search(r, line)
            if a is not None:
                pattern = r"<ConvTypes.*conv'>,"
                to_remove = re.findall(pattern, line, flags=0)
                pattern_2 = r"'[A-Z,a-z]+conv'"
                to_keep = re.findall(pattern_2, to_remove[0], flags=0)
                mod_string = re.sub(to_remove[0], to_keep[0]+',', line )
                pattern = r"<LossTypes.*'>,"
                to_remove = re.findall(pattern, line, flags=0)
                pattern_2 = r"'[A-Z,a-z]+'"
                to_keep = re.findall(pattern_2, to_remove[0], flags=0)

                mod_string = re.sub(to_remove[0], to_keep[0]+',', mod_string )
                d_name = eval(mod_string)
                
                if d_name['variational'] == "True":
                    v= "VAR"
                else:
                    v= "GAE"
                name = (name +
                        v+
                        "_"+str(d_name['convtype']) +
                        "_"+str(d_name['batchsize']) +
                        "_"+str(d_name['latentdim']) +
                        "_"+str(d_name['nhidden']) +
                        "_"+str(d_name['convenc']) +
                        "_"+str(d_name['denseenc']) +
                        "_"+str(d_name['densedeca']) +
                        "_"+str(d_name['convdecx']) +
                        "_"+str(d_name['densedecx']) +
                        "_id_"+str(d_name['trainid']) 
                        )
                return name

    return "None"

def get_last_epoch_loss(file_name):
    loss = "0"
    epoch = "0"
    rEpoch = "EPOCH*"
    rLoss = "Loss train:*"
    with open(file_name, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)-1,0,-1):
            a1 = re.search(rLoss, lines[i])
            if a1 is not None:
                a2 = re.search(rEpoch, lines[i-1])
                if a2 is not None:
                    #print(lines[i-1])
                    #exit(0)
                    epoch = re.findall('[0-9]+', lines[i-1])[0]
                    loss = re.findall('[0-9]+.[0-9]+', lines[i])[0]
                    return epoch,loss
    return epoch,loss

class options(Enum):
    FILE = 'file'
    LOSS = 'loss'
    EPOCH = 'epoch'

def parseArguments():
    parser = argparse.ArgumentParser(
        description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[
            0])
    parser.add_argument('-order', type=options, required=False, help='Order by: file, loss or epoch. Default  file',default=options.FILE)


    parser.set_defaults(debug=False)
    return parser.parse_args()

parsed = parseArguments()

file_list = sorted(create_file_list())
files_dict ={}
none_count= 0
for file in file_list:
    name = get_file_name(file)

    epoch,loss = get_last_epoch_loss(file)
    if parsed.order == options.FILE:
        files_dict[name] = epoch + " : " + loss
    elif parsed.order == options.LOSS:
        files_dict[loss] = name + " : " + epoch
    elif parsed.order == options.EPOCH:
        files_dict[epoch] = name + " : " + loss
    # print(name, "epoch:",epoch," loss:",loss)
    if name == "None":
        none_count +=1
if parsed.order == options.FILE:
    pprint(files_dict)
else:
    pprint(sorted((float(x),y) for x,y in files_dict.items()))
print("WRONG FILES: ",none_count)
# pprint(files_dict)
# for key, value in files_dict: 
#     print("{} : {}".format(key, value))

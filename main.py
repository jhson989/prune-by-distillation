
import argparse, os
import random
import torch
from utils import Logger
from trainer import Trainer
#############################################################
# Hyper-parameters
#############################################################
import easydict
args = easydict.EasyDict({ 
    "train" : True, 
    
    # Train policy
    "numEpoch" : 1000,
    "batchSize" : 8,
    "lr" : 1e-4,
    "manualSeed" : 1,

    # Record
    "savePath" : "./result/unpruned/",
    "retrain" : True, 
    "loadPath" : "./result/unpruned/best.pth",
    "logFreq" : 5,   

    # Hardware
    "ngpu" : 1,
    "numWorkers" : 5,    
})

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

try:
    if not os.path.exists(args.savePath):
        os.makedirs(args.savePath)
except OSError:
    print("Error: Creating save folder. [" + args.savePath + "]")

if torch.cuda.is_available() == False:
    args.ngpu = 0

if args.ngpu == 1:
    args.device = torch.device("cuda")
else :
    args.device = torch.device("cpu")


logger = Logger(args.savePath, args.retrain)
logger.log(str(args))

logger.log("[[[Train]]] Train started..")

# Define trainer
trainer = Trainer(args=args, logger=logger)

# Start training
trainer.train()
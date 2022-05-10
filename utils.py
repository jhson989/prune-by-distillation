import numpy as np
import os
import torch
from PIL import Image

class Logger():
    def __init__(self, path, retrain):
        self.logFile = None
        if os.path.isfile(path+"log.txt") and retrain == True:
            self.logFile = open(path+"log.txt", "a")
            self.logFile.write("\n\n\n [[[[Retrain]]]] \n")
        else :
            self.logFile = open(path+"log.txt", "w")
        
    def __del__(self):
        self.logFile.close()

    def log(self, logStr):
        print(logStr)
        self.logFile.write(logStr+"\n")
        self.logFile.flush()


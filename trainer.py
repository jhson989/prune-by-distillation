import torch
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch import nn

from data import getDataLoader
from model import getPretrainedModel

class Trainer:

    def __init__(self, args, logger):

        # Argument
        self.args = args
        self.logger = logger

        ### Data
        self.dataLoader = getDataLoader(train=True, args=args)
        self.evalDataLoader = getDataLoader(train=False, args=args)

        # Model
        self.teacher = getPretrainedModel(pretrained=True).to(self.args.device)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        self.student = getPretrainedModel(pretrained=True).to(self.args.device)
        
        # Criterion
        self.optimizer = optim.Adam(self.student.parameters(), lr=self.args.lr)
        self.critTeacher = nn.MSELoss().to(self.args.device)

        # Prune
        self.pruned = False
        self.lastBestLoss = 5.0
        self.currentBestLoss = 100.0
        self.recentLosses = [100.0]
        self.pruneAmount = 0.01

        # Save
        self.bestLoss = 100.0



    def train(self):

        ### Re-training
        startEpoch = 0
        if self.args.retrain == True:
            startEpoch = self.load(self.args.loadPath)
        self.pruning()

        ### Training iteration
        for epoch in range(startEpoch, self.args.numEpoch):

            ### Train
            self.student.train()
            for idx, (img, _) in enumerate(self.dataLoader):
                ### data
                img = img.to(self.args.device)

                ### learning
                self.optimizer.zero_grad()
                predTeacher = self.teacher(img)['out']
                pred = self.student(img)['out']
                loss = self.critTeacher(pred, predTeacher)
                loss.backward()
                self.optimizer.step()

                ### Logging
                if idx % self.args.logFreq == 0: 
                    self.logger.log(
                        "[[%4d/%4d] [%4d/%4d]] loss : loss(%.3f)"  
                        % (epoch, self.args.numEpoch, idx, len(self.dataLoader), loss.item())
                    )

            ### Eval
            avgLoss = 0.0
            if self.evalDataLoader is not None :
                avgLoss = self.eval(self.evalDataLoader, epoch)

            self.adaptivePruning(avgLoss)



    def eval(self, evalDataLoader, epoch):

        ### Eval
        avgLoss = 0.0

        self.student.eval()
        with torch.no_grad():            
            for idx, (img, _) in enumerate(evalDataLoader):
                img = img.to(self.args.device)
                predTeacher = self.teacher(img)['out']
                pred = self.student(img)['out']
                loss = self.critTeacher(pred, predTeacher)
                avgLoss = avgLoss + loss.item()

        ### Logging
        avgLoss = avgLoss/len(evalDataLoader)
        self.logger.log("Eval loss : loss(%.3f)" % (avgLoss))

        if avgLoss < self.bestLoss :
            self.bestLoss = avgLoss
            self.save("best.pth", epoch)
        
        self.save("last.pth", epoch)
        return avgLoss



    def adaptivePruning(self, loss, searchRegion=10):
        
        self.recentLosses.append(loss)
        self.recentLosses = self.recentLosses[1:] if len(self.recentLosses) > searchRegion else self.recentLosses
        
        self.currentBestLoss = loss if loss < self.currentBestLoss else self.currentBestLoss

        if sum(int(l<self.lastBestLoss) for l in self.recentLosses) > int(searchRegion * 0.2):

            self.lastBestLoss = self.currentBestLoss
            self.recentLosses = [self.recentLosses[-1]]
            self.pruneAmount = self.pruneAmount + 0.01
            self.remove()
            self.pruning()
            self.printInfo()



    def pruning(self):

        if self.pruned == True:
            return

        self.pruned = True
        for name, module in self.student.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.pruneAmount)



    def remove(self):

        if self.pruned == False:
            return

        self.pruned = False
        for name, module in self.student.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.remove(module, 'weight')



    def printInfo(self):
        ## total number of parameters
        numParams = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        numNonzeros = sum(torch.count_nonzero(p) for p in self.student.parameters() if p.requires_grad)
        self.logger.log("Pruning ratio : %d/%d = (%.3f/%.3f)GB =  %.3f %%" % 
                (numNonzeros, numParams, float(numNonzeros)*8/pow(2,30), float(numParams)*8/pow(2,30), float(numNonzeros)/numParams*100)
              )



    def save(self, filename, numEpoch):

        filename = self.args.savePath + filename

        self.remove()
        self.student.eval()
        torch.save({
            "epoch" : numEpoch,
            "studentStateDict" : self.student.state_dict(),
            "optimizerStateDict" : self.optimizer.state_dict(),
            # Prune
            "prune" : [
                self.lastBestLoss,
                self.currentBestLoss,
                self.recentLosses,
                self.pruneAmount
            ]
            }, filename)
        self.pruning()



    def load(self, filename):

        checkpoint = torch.load(filename)
        self.student.load_state_dict(checkpoint["studentStateDict"])
        self.optimizer.load_state_dict(checkpoint["optimizerStateDict"])

        # Prune
        pruneList = checkpoint["prune"]
        self.lastBestLoss = pruneList[0]
        self.currentBestLoss = pruneList[1]
        self.recentLosses = pruneList[2]
        self.pruneAmount = pruneList[3]

        return checkpoint["epoch"]
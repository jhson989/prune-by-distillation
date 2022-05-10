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
        self.student = getPretrainedModel(pretrained=False).to(self.args.device)
        

        # Criterion
        self.optimizer = optim.Adam(self.student.parameters(), lr=self.args.lr)
        self.critTeacher = nn.MSELoss().to(self.args.device)
        self.critGT = nn.MSELoss().to(self.args.device)

        # Save
        self.bestLoss = 100.0


    def train(self):

        ### Re-training
        startEpoch = 0
        if self.args.retrain == True:
            startEpoch = self.load(self.args.loadPath)

        ### Training iteration
        for epoch in range(startEpoch, self.args.numEpoch):

            ### Train
            self.student.train()
            for idx, (img, gt) in enumerate(self.dataLoader):
                ### data
                #img, gt = img.to(self.args.device), gt.to(self.args.device)
                img = img.to(self.args.device)

                ### learning
                self.optimizer.zero_grad()
                predTeacher = self.teacher(img)['out']
                pred = self.student(img)['out']
                lossTeacher = self.critTeacher(pred, predTeacher)
                #lossGT = self.critGT(pred.argmax(dim=1), gt)
                loss = lossTeacher# + lossGT
                loss.backward()
                self.optimizer.step()

                ### Logging
                if idx % self.args.logFreq == 0: 
                    self.logger.log(
                        "[[%4d/%4d] [%4d/%4d]] loss : Total(%.3f) = Teacher(%.3f) + GT(%.3f)"  
                        % (epoch, self.args.numEpoch, idx, len(self.dataLoader), loss.item(), lossTeacher.item(), 0 )
                    )
                    
            ### Eval
            if self.evalDataLoader is not None :
                self.eval(self.evalDataLoader, epoch)
    


    def eval(self, evalDataLoader, epoch):

        ### Eval
        avgLoss = 0.0
        self.student.eval()

        with torch.no_grad():            
            for idx, (img, gt) in enumerate(evalDataLoader):
                img, gt = img.to(self.args.device), gt.to(self.args.device)
                predTeacher = self.teacher(img)['out']
                pred = self.student(img)['out']
                loss = self.critTeacher(pred, predTeacher)# + self.critGT(pred, gt)
                avgLoss = avgLoss + loss.item()

        ### Logging
        avgLoss = avgLoss/len(evalDataLoader)
        self.logger.log("Eval loss : Total(%.3f)" % (avgLoss))

        if avgLoss < self.bestLoss :
            self.logger.log("Best model at %d" % epoch)
            self.bestLoss = avgLoss
            self.save("best.pth", epoch)
        
        self.save("last.pth", epoch)
        return avgLoss


    def save(self, filename, numEpoch):

        filename = self.args.savePath + filename

        self.student.eval()
        torch.save({
            "epoch" : numEpoch,
            "studentStateDict" : self.student.state_dict(),
            "optimizerStateDict" : self.optimizer.state_dict(),
            }, filename)



    def load(self, filename):

        checkpoint = torch.load(filename)
        self.student.load_state_dict(checkpoint["studentStateDict"])
        self.optimizer.load_state_dict(checkpoint["optimizerStateDict"])

        return checkpoint["epoch"]
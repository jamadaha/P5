from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import BaseKerasModelTrainer as baseKeras

import time
from ProjectTools import CSVLogger
from ProjectTools import TFLogger

class ClassifierTrainer(baseKeras.BaseKerasModelTrainer):
    Logger = None
    SummaryWriter = None

    __latestLoss = 0
    __latestAccuracy = 0

    def __init__(self, model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath, logPath):
        super().__init__(model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath)
        self.Logger = CSVLogger.CSVLogger(logPath, 'TrainingData')
        self.Logger.InitCSV(['Epoch', 'Loss', 'Accuracy'])
        self.SummaryWriter = {
            'CLoss': TFLogger.TFLogger(logPath, 'Loss', 'CLoss'),
            'Accuracy': TFLogger.TFLogger(logPath, 'Accuracy', 'Classifier')
        }

    def PrintStatus(self, iteration, totalIterations, epochTime, epoch):
        estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
        print(f"Classifier loss: {self.__latestLoss:.4f}. Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")

    def PrintTestStatus(self, iteration, totalIterations, epochTime):
        estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
        print(f"Accuracy: {(self.__latestAccuracy*100):.2f}% Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")
        
    def LogData(self, epoch):
        self.Logger.AppendToCSV([epoch + 1, self.__latestLoss, self.__latestAccuracy])
        self.SummaryWriter['CLoss'].LogNumber(self.__latestLoss, epoch + 1)
        self.SummaryWriter['Accuracy'].LogNumber(self.__latestAccuracy, epoch + 1)

    def SetTrainProperties(self, returnVal):
        self.__latestLoss = float(returnVal['c_loss'])

    def SetTestProperties(self, returnTest):
        self.__latestAccuracy = float(returnTest['classifier_accuracy'])

    def EpochRun(self, epoch):
        super().EpochRun(epoch)
        self.Model.AccuracyTracker.reset_state()

    def SaveCheckpoint(self, ckptPath):
        self.Model.Classifier.save_weights(ckptPath)
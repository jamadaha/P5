from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import BaseKerasModelTrainer as baseKeras

import time
from ProjectTools import CSVLogger
from ProjectTools import TFLogger
from CGAN import CGANKerasModel, LetterProducer

class CGANTrainer(baseKeras.BaseKerasModelTrainer):
    Logger = None
    SummaryWriter = None
    LetterProducer = None

    __latestGLoss = 0
    __latestDLoss = 0
    __ModeCollapseValue = 0

    TrackModeCollapse = False
    ModeCollapseThreshold = 0

    def __init__(self, model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath, logPath, numberOfClasses, latentDimension, epochImgDir, trackModeCollapse, modeCollapseThreshold):
        super().__init__(model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath)
        self.Logger = CSVLogger.CSVLogger(logPath, 'TrainingData')
        self.Logger.InitCSV(['Epoch', 'GeneratorLoss', 'DiscriminatorLoss', 'ImageUniqueness'])
        self.SummaryWriter = {
            'GLoss': TFLogger.TFLogger(logPath, 'Loss', 'GLoss'),
            'DLoss': TFLogger.TFLogger(logPath, 'Loss', 'DLoss'),
            'ModeCollapseValue': TFLogger.TFLogger(logPath, 'ModeCollapse', 'GAN'),
            'DiffLoss': TFLogger.TFLogger(logPath, 'Loss', 'DiffLoss'),
            'Images': TFLogger.TFLogger(logPath, '', 'Images')
        }
        self.LetterProducer = LetterProducer.LetterProducer(epochImgDir, self.Model.Generator, numberOfClasses, latentDimension, 0)
        self.ModeCollapseThreshold = modeCollapseThreshold
        self.TrackModeCollapse = trackModeCollapse
           
    def PrintStatus(self, iteration, totalIterations, epochTime, epoch):
        estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
        print(f"Generator loss: {self.__latestGLoss:.4f}. Discriminator loss: {self.__latestDLoss:.4f}. Is in mode collapse?: {self.GetModeCollapseValue()}. Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}              ", end="\r")

    def GetModeCollapseValue(self):
        if self.TrackModeCollapse == True:
            if self.__ModeCollapseValue <= self.ModeCollapseThreshold:
                return f"POTENTIAL ({self.__ModeCollapseValue:.2f})"
            return "NO"
        else:
            return "Not Tracking"

    def LogData(self, epoch):
        print("Logging and generating epoch images...")
        self.Logger.AppendToCSV([epoch + 1, self.__latestGLoss, self.__latestDLoss, self.__ModeCollapseValue])

        self.SummaryWriter['GLoss'].LogNumber(self.__latestGLoss, epoch + 1)
        self.SummaryWriter['DLoss'].LogNumber(self.__latestDLoss, epoch + 1)
        self.SummaryWriter['ModeCollapseValue'].LogNumber(self.__ModeCollapseValue, epoch + 1)
        self.SummaryWriter['DiffLoss'].LogNumber(abs(self.__latestDLoss - self.__latestGLoss), epoch + 1)
        self.SummaryWriter['Images'].LogGridImages(self.ProduceGridImage(epoch + 1), epoch + 1)

    def SetTrainProperties(self, returnVal):
        self.__latestGLoss = float(returnVal['g_loss'])
        self.__latestDLoss = float(returnVal['d_loss'])
        self.__ModeCollapseValue = float(returnVal['mode_collapse_loss'])

    def GenerateSampleImages(self):
        return self.LetterProducer.GetSampleLetters()

    def ProduceGridImage(self, id):
        return self.LetterProducer.ProduceGridLetters(id)

    def SaveCheckpoint(self, ckptPath):
        self.Model.Generator.save_weights(ckptPath)
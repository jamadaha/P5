from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import BaseKerasModelTrainer as baseKeras

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")

import tensorflow as tf
from tensorflow import keras
import time
import os
from ProjectTools import CSVLogger
from ProjectTools import TFLogger
from CGAN import LetterProducer
import tensorboard

class CGANTrainer(baseKeras.BaseKerasModelTrainer):
    Logger = None
    SummaryWriter = None
    LetterProducer = None

    __latestGLoss = 0
    __latestDLoss = 0

    def __init__(self, model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath, logPath, numberOfClasses, latentDimension):
        super().__init__(model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath)
        self.Logger = CSVLogger.CSVLogger(logPath, 'TrainingData')
        self.Logger.InitCSV(['Epoch', 'GeneratorLoss', 'DiscriminatorLoss'])
        self.SummaryWriter = {
            'GLoss': TFLogger.TFLogger(logPath, 'Loss', 'GLoss'),
            'DLoss': TFLogger.TFLogger(logPath, 'Loss', 'DLoss'),
            'DiffLoss': TFLogger.TFLogger(logPath, 'Loss', 'DiffLoss'),
            'Images': TFLogger.TFLogger(logPath, '', 'Images')
        }
        self.LetterProducer = LetterProducer.LetterProducer('', self.Model.generator, numberOfClasses, latentDimension, 0)
           
    def PrintStatus(self, iteration, totalIterations, epochTime, epoch):
        estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
        print(f"Generator loss: {self.__latestGLoss:.4f}. Discriminator loss: {self.__latestDLoss:.4f}. Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")

    def LogData(self, epoch):
        self.Logger.AppendToCSV([epoch + 1, self.__latestGLoss, self.__latestDLoss])

        self.SummaryWriter['GLoss'].LogNumber(self.__latestGLoss, epoch + 1)
        self.SummaryWriter['DLoss'].LogNumber(self.__latestDLoss, epoch + 1)
        self.SummaryWriter['DiffLoss'].LogNumber(abs(self.__latestDLoss - self.__latestGLoss), epoch + 1)
        self.SummaryWriter['Images'].LogGridImages(self.GenerateSampleImages(), epoch + 1)

    def SetTrainProperties(self, returnVal):
        self.__latestGLoss = float(returnVal['g_loss'])
        self.__latestDLoss = float(returnVal['d_loss'])

    def GenerateSampleImages(self):
        return self.LetterProducer.GetSampleLetters()

from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import BaseKerasModelTrainer as baseKeras

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")

import tensorflow as tf
from tensorflow import keras
import time
import os
from ProjectTools import Logger as lgr

class ClassifierTrainer(baseKeras.BaseKerasModelTrainer):
    Logger = None
    SummaryWriter = None

    __latestLoss = 0
    __latestAccuracy = 0

    def __init__(self, model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath, logPath):
        super().__init__(model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath)
        self.Logger = lgr.Logger(logPath, 'TrainingData')
        self.Logger.InitCSV(['Epoch', 'Loss', 'Accuracy'])
        self.SummaryWriter = {
            'CLoss': tf.summary.create_file_writer(os.path.join(logPath, 'Loss', 'CLoss')),
            'Accuracy': tf.summary.create_file_writer(os.path.join(logPath, 'Accuracy'))
        }

    def PrintStatus(self, iteration, totalIterations, epochTime, epoch):
        estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
        print(f"Classifier loss: {self.__latestLoss:.4f}. Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")

    def PrintTestStatus(self, iteration, totalIterations, epochTime):
        estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
        print(f"Accuracy: {(self.__latestAccuracy*100):.2f}% Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")
        
    def LogData(self, epoch):
        self.Logger.AppendToCSV([epoch + 1, self.__latestLoss, self.__latestAccuracy])

        with self.SummaryWriter['CLoss'].as_default():
            with tf.name_scope('Loss'):
                tf.summary.scalar('ClassifierLoss', self.__latestLoss, step=epoch)
        with self.SummaryWriter['Accuracy'].as_default():
            with tf.name_scope('Accuracy'):
                tf.summary.scalar('ClassifierAccuracy', self.__latestAccuracy, step=epoch)

    def SetTrainProperties(self, returnVal):
        self.__latestLoss = float(returnVal['c_loss'])

    def SetTestProperties(self, returnTest):
        self.__latestAccuracy = float(returnTest['classifier_accuracy'])
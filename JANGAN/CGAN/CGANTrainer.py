from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import BaseKerasModelTrainer as baseKeras

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")

import tensorflow as tf
from tensorflow import keras
import time
import os
from ProjectTools import Logger as lgr

class CGANTrainer(baseKeras.BaseKerasModelTrainer):
    Logger = None
    SummaryWriter = None

    __latestGLoss = 0
    __latestDLoss = 0

    def __init__(self, model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath, logPath):
        super().__init__(model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath)
        self.Logger = lgr.Logger(logPath, 'TrainingData')
        self.Logger.InitCSV(['Epoch', 'GeneratorLoss', 'DiscriminatorLoss'])
        self.SummaryWriter = {
            'GLoss': tf.summary.create_file_writer(os.path.join(logPath, 'Loss', 'GLoss')),
            'DLoss': tf.summary.create_file_writer(os.path.join(logPath, 'Loss', 'DLoss')),
            'DiffLoss': tf.summary.create_file_writer(os.path.join(logPath, 'Loss', 'DiffLoss')),
        }
           
    def PrintStatus(self, iteration, totalIterations, epochTime, epoch):
        estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
        print(f"Generator loss: {self.__latestGLoss:.4f}. Discriminator loss: {self.__latestDLoss:.4f}. Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")

    def LogData(self, epoch):
        self.Logger.AppendToCSV([epoch + 1, self.__latestGLoss, self.__latestDLoss])

        with self.SummaryWriter['GLoss'].as_default():
            with tf.name_scope('Loss'):
                tf.summary.scalar('Loss', self.__latestGLoss, step=epoch)
        with self.SummaryWriter['DLoss'].as_default():
            with tf.name_scope('Loss'):
                tf.summary.scalar('Loss', self.__latestDLoss, step=epoch)
        with self.SummaryWriter['DiffLoss'].as_default():
            with tf.name_scope('Loss'):
                tf.summary.scalar('DiffLoss', abs(self.__latestDLoss - self.__latestGLoss), step=epoch)

    def SetTrainProperties(self, returnVal):
        self.__latestGLoss = float(returnVal['g_loss'])
        self.__latestDLoss = float(returnVal['d_loss'])
from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")

import tensorflow as tf
import time
import os
import csv
from ProjectTools import Logger as lgr

class CGANTrainer():
    CGAN = None
    Datasets = []
    Epochs = 0
    RefreshUIEachXStep = 1
    SaveCheckpoints = False
    CheckpointPath = ""
    Logger = None

    __latestGLoss = 0
    __latestDLoss = 0

    def __init__(self, cGAN, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, logPath):
        self.CGAN = cGAN
        self.Datasets = datasets
        self.Epochs = epochs
        self.RefreshUIEachXStep = refreshUIEachXStep
        self.SaveCheckpoints = saveCheckPoints
        self.CheckpointPath = checkpointPath
        self.Logger = lgr.Logger(logPath, 'TrainingData')
        self.Logger.InitCSV(['Epoch', 'GeneratorLoss', 'DiscriminatorLoss'])

    def TrainCGAN(self):
        print("Training started")
        for epoch in range(self.Epochs):
            start = time.time()

            print(f"Epoch {epoch + 1} of {self.Epochs} is in progress...")
            
            self.__EpochRun(epoch)

            totalEpochTime = time.time()-start
            print("")
            print("Done!")
            print(f"Time for epoch {epoch + 1} is {self.GetDatetimeFromSeconds(totalEpochTime)}. Est time remaining for training is {self.GetDatetimeFromSeconds(totalEpochTime*(self.Epochs-(epoch + 1)))}")

            if self.SaveCheckpoints:
                self.__SaveCheckpoint()
        print("Training finished!")
            
    def CreateDataSet(self, dataArray):
        returnSet = dataArray[0]
        for data in dataArray[1:]:
            returnSet = returnSet.concatenate(data)
        return returnSet.shuffle(buffer_size=1024)

    def GetDatetimeFromSeconds(self, seconds):
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    def __PrintStatus(self, iteration, totalIterations, epochTime, epoch):
        estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
        print(f"Generator loss: {self.__latestGLoss:.4f}. Discriminator loss: {self.__latestDLoss:.4f}. Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")
        self.Logger.AppendToCSV([epoch + 1, self.__latestGLoss, self.__latestDLoss])

    def __SaveCheckpoint(self):
        if os.path.exists(self.CheckpointPath + 'cgan_checkpoint.index'):
            from ProjectTools import HelperFunctions as hf
            hf.DeleteFolderAndAllContents(self.CheckpointPath)
        self.CGAN.save_weights(self.CheckpointPath + 'cgan_checkpoint')

    def __EpochRun(self, epoch):
        epochDataset = self.CreateDataSet(self.Datasets)
        totalIterations = tf.data.experimental.cardinality(epochDataset).numpy()
        iteration = 0
        epochTime = time.time()
        for image_batch in epochDataset:
            if iteration % self.RefreshUIEachXStep == 0:
                returnVal = self.CGAN.train_step(image_batch, True)
                self.__latestGLoss = float(returnVal['g_loss'])
                self.__latestDLoss = float(returnVal['d_loss'])
                self.__PrintStatus(iteration, totalIterations, epochTime, epoch)
                epochTime = time.time()
            else:
                self.CGAN.train_step(image_batch, False)
            iteration += 1
        self.__PrintStatus(totalIterations, totalIterations, epochTime, epoch)

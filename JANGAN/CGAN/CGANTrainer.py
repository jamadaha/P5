from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")
ap.CheckAndInstall("csv")

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
            epochDataset = self.CreateDataSet(self.Datasets)
            itemCount = tf.data.experimental.cardinality(epochDataset).numpy()
            count = 0
            epochTime = time.time()
            for image_batch in epochDataset:
                if count % self.RefreshUIEachXStep == 0:
                    returnVal = self.CGAN.train_step(image_batch)
                    g_loss = float(returnVal['g_loss'])
                    d_loss = float(returnVal['d_loss'])
                    now = time.time()
                    estRemainTime = ((now - epochTime) / self.RefreshUIEachXStep) * (itemCount - count)
                    epochTime = now
                    print(f"Generator loss: {g_loss:.4f}. Discriminator loss: {d_loss:.4f}. Progress: {((count/itemCount)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainTime)}    ", end="\r")
                    self.Logger.AppendToCSV([epoch + 1, g_loss, d_loss])
                else:
                    self.CGAN.train_step(image_batch)
                count += 1

            totalEpochTime = time.time()-start
            print("")
            print("Done!")
            print(f"Time for epoch {epoch + 1} is {self.GetDatetimeFromSeconds(totalEpochTime)}. Est time remaining for training is {self.GetDatetimeFromSeconds(totalEpochTime*(self.Epochs-(epoch + 1)))}")

            if self.SaveCheckpoints:
                if os.path.exists(self.CheckpointPath + 'cgan_checkpoint.index'):
                    from ProjectTools import HelperFunctions as hf
                    hf.DeleteFolderAndAllContents(self.CheckpointPath)
                self.CGAN.save_weights(self.CheckpointPath + 'cgan_checkpoint')
            
    def CreateDataSet(self, dataArray):
        returnSet = dataArray[0]
        for data in dataArray[1:]:
            returnSet = returnSet.concatenate(data)
        return returnSet.shuffle(buffer_size=1024)

    def GetDatetimeFromSeconds(self, seconds):
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

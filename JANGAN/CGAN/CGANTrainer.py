from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")

import tensorflow as tf
from tensorflow import keras
import time
import os
from ProjectTools import CSVLogger
from ProjectTools import TFLogger
from CGAN import LetterProducer as lp
import tensorboard

class CGANTrainer():
    CGAN = None
    Datasets = []
    Epochs = 0
    CurrentEpoch = None
    RefreshUIEachXStep = 1
    SaveCheckpoints = False
    CheckpointPath = ""
    LatestCheckpointPath = ""
    LetterProducer = None
    Logger = None
    SummaryWriter = None

    __latestGLoss = 0
    __latestDLoss = 0

    def __init__(self, cGAN, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath, logPath, numberOfClasses, latentDimension):
        self.CGAN = cGAN
        self.Datasets = datasets
        self.Epochs = epochs
        self.RefreshUIEachXStep = refreshUIEachXStep
        self.SaveCheckpoints = saveCheckPoints
        self.CheckpointPath = checkpointPath
        self.LatestCheckpointPath = latestCheckpointPath
        self.Logger = CSVLogger.CSVLogger(logPath, 'TrainingData')
        self.Logger.InitCSV(['Epoch', 'GeneratorLoss', 'DiscriminatorLoss'])
        self.SummaryWriter = {
            'GLoss': TFLogger.TFLogger(logPath, 'Loss', 'GLoss'),
            'DLoss': TFLogger.TFLogger(logPath, 'Loss', 'DLoss'),
            'DiffLoss': TFLogger.TFLogger(logPath, 'Loss', 'DiffLoss'),
            'Images': TFLogger.TFLogger(logPath, '', 'Images'),
        }
        self.LetterProducer = lp.LetterProducer('', self.CGAN.generator, numberOfClasses, latentDimension, 0)


    def TrainCGAN(self):
        print("Training started")
        for epoch in range(self.Epochs):
            self.CurrentEpoch = epoch
            start = time.time()

            print(f"Epoch {epoch + 1} of {self.Epochs} is in progress...")
            
            self.__EpochRun(epoch)

            totalEpochTime = time.time()-start

            print(f"Time for epoch {epoch + 1} is {self.GetDatetimeFromSeconds(totalEpochTime)}. Est time remaining for training is {self.GetDatetimeFromSeconds(totalEpochTime*(self.Epochs-(epoch + 1)))}")
        
        print("Training finished!")
            
    def CreateDataSet(self, dataArray):
        (returnTrainSet, returnTestSet) = dataArray[0]
        for data in dataArray[1:]:
            (addTrainSet, addTestSet) = data
            returnTrainSet = returnTrainSet.concatenate(addTrainSet)
            returnTestSet = returnTestSet.concatenate(addTestSet)
        
        returnTrainSet = returnTrainSet.shuffle(buffer_size=1024)
        returnTestSet = returnTestSet.shuffle(buffer_size=1024)
        return (returnTrainSet, returnTestSet)

    def GetDatetimeFromSeconds(self, seconds):
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    def __PrintStatus(self, iteration, totalIterations, epochTime, epoch):
        estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
        print(f"Generator loss: {self.__latestGLoss:.4f}. Discriminator loss: {self.__latestDLoss:.4f}. Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")

    def __SaveCheckpoint(self):
        if os.path.exists(self.CheckpointPath + 'cgan_checkpoint.index'):
            from ProjectTools import HelperFunctions as hf
            hf.DeleteFolderAndAllContents(self.CheckpointPath)

        ckptPath = self.CheckpointPath + 'cgan_checkpoint_' + str(self.CurrentEpoch)

        self.CGAN.save_weights(ckptPath)
        self.__MakeCheckpointRef(ckptPath, self.LatestCheckpointPath)

    def __MakeCheckpointRef(self, ckptPathSrc, ckptPathDest):
        relPath = os.path.relpath(os.path.abspath(ckptPathSrc), os.path.abspath(ckptPathDest))
        os.makedirs(os.path.dirname(ckptPathDest), exist_ok=True)
        with open(ckptPathDest, 'w') as f:
            f.write(f"{ckptPathSrc}")

    def __LogData(self, images, epoch):
        self.Logger.AppendToCSV([epoch + 1, self.__latestGLoss, self.__latestDLoss])

        self.SummaryWriter['GLoss'].LogNumber(self.__latestGLoss, epoch + 1)
        self.SummaryWriter['DLoss'].LogNumber(self.__latestDLoss, epoch + 1)
        self.SummaryWriter['DiffLoss'].LogNumber(abs(self.__latestDLoss - self.__latestGLoss), epoch + 1)
        self.SummaryWriter['Images'].LogGridImages(images, epoch + 1)

    def __EpochRun(self, epoch):
        print("Training CGAN...")
        (image_batch_train, image_batch_test) = self.CreateDataSet(self.Datasets)
        totalIterations = tf.data.experimental.cardinality(image_batch_train).numpy()
        iteration = 0
        epochTime = time.time()
        for image_batch in image_batch_train:
            if iteration % self.RefreshUIEachXStep == 0:
                returnVal = self.CGAN.train_step(image_batch, True)
                self.__latestGLoss = float(returnVal['g_loss'])
                self.__latestDLoss = float(returnVal['d_loss'])
                self.__PrintStatus(iteration, totalIterations, epochTime, epoch)
                epochTime = time.time()
            else:
                self.CGAN.train_step(image_batch, False)
            iteration += 1
        
        keras.backend.clear_session()

        self.__PrintStatus(totalIterations, totalIterations, epochTime, epoch)
        print("")
        print("Done!")

        if self.SaveCheckpoints:
            self.__SaveCheckpoint()

        images = self.LetterProducer.GetSampleLetters()
        self.__LogData(images, epoch)

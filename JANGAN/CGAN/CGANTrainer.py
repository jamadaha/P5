from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")

import tensorflow as tf
from tensorflow import keras
import time
import os
from ProjectTools import Logger as lgr
import tensorboard

class CGANTrainer():
    CGAN = None
    Datasets = []
    Epochs = 0
    RefreshUIEachXStep = 1
    SaveCheckpoints = False
    CheckpointPath = ""
    Logger = None
    SummaryWriter = None

    __latestGLoss = 0
    __latestDLoss = 0
    #__latestAccuracy = 0

    def __init__(self, cGAN, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, logPath):
        self.CGAN = cGAN
        self.Datasets = datasets
        self.Epochs = epochs
        self.RefreshUIEachXStep = refreshUIEachXStep
        self.SaveCheckpoints = saveCheckPoints
        self.CheckpointPath = checkpointPath
        self.Logger = lgr.Logger(logPath, 'TrainingData')
        self.Logger.InitCSV(['Epoch', 'GeneratorLoss', 'DiscriminatorLoss'])
        self.SummaryWriter = {
            'GLoss': tf.summary.create_file_writer(os.path.join(logPath, 'Loss', 'GLoss')),
            'DLoss': tf.summary.create_file_writer(os.path.join(logPath, 'Loss', 'DLoss')),
            'DiffLoss': tf.summary.create_file_writer(os.path.join(logPath, 'Loss', 'DiffLoss')),
            #'Accuracy': tf.summary.create_file_writer(os.path.join(logPath, 'Accuracy'))
        }

    def TrainCGAN(self):
        print("Training started")
        for epoch in range(self.Epochs):
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

    #def __PrintTestStatus(self, iteration, totalIterations, epochTime):
    #    estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
    #    print(f"Accuracy: {(self.__latestAccuracy*100):.2f}% Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")

    def __SaveCheckpoint(self):
        if os.path.exists(self.CheckpointPath + 'cgan_checkpoint.index'):
            from ProjectTools import HelperFunctions as hf
            hf.DeleteFolderAndAllContents(self.CheckpointPath)
        self.CGAN.save_weights(self.CheckpointPath + 'cgan_checkpoint')

    def __LogData(self, epoch):
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
        #with self.SummaryWriter['Accuracy'].as_default():
        #    with tf.name_scope('Accuracy'):
        #        tf.summary.scalar('Accuracy', self.__latestAccuracy, step=epoch)

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

        #print("Testing CGAN...")
        #iteration = 0
        #for image_batch in image_batch_test:
        #    if iteration % self.RefreshUIEachXStep == 0:
        #        returnTest = self.CGAN.test_step(image_batch, True)
        #        self.__latestAccuracy = float(returnTest['cgan_accuracy'])
        #        self.__PrintTestStatus(iteration, totalIterations, epochTime)
        #        epochTime = time.time()
        #    else:
        #        self.CGAN.test_step(image_batch, False)
        #    iteration += 1
        #self.CGAN.CGANAccuracy_tracker.reset_state()

        #self.__PrintTestStatus(totalIterations, totalIterations, epochTime)
        #print("")
        #print("Done!")

        if self.SaveCheckpoints:
            self.__SaveCheckpoint()

        self.__LogData(epoch)

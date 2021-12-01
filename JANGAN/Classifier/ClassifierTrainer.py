from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")

import tensorflow as tf
from tensorflow import keras
import time
import os
import shutil
from ProjectTools import CSVLogger
from ProjectTools import TFLogger
import tensorboard

class ClassifierTrainer():
    Classifier = None
    Datasets = []
    Epochs = 0
    CurrentEpoch = None
    RefreshUIEachXStep = 1
    SaveCheckpoints = False
    CheckpointPath = ""
    LatestCheckpointPath = ""
    Logger = None
    SummaryWriter = None

    __latestLoss = 0
    __latestAccuracy = 0

    def __init__(self, classifier, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath, logPath):
        self.Classifier = classifier
        self.Datasets = datasets
        self.Epochs = epochs
        self.RefreshUIEachXStep = refreshUIEachXStep
        self.SaveCheckpoints = saveCheckPoints
        self.CheckpointPath = checkpointPath
        self.LatestCheckpointPath = latestCheckpointPath
        self.Logger = CSVLogger.CSVLogger(logPath, 'TrainingData')
        self.Logger.InitCSV(['Epoch', 'GeneratorLoss', 'DiscriminatorLoss'])
        self.SummaryWriter = {
            'CLoss': TFLogger.TFLogger(logPath, 'Loss', 'CLoss'),
            'Accuracy': TFLogger.TFLogger(logPath, 'Classifier', 'Accuracy')
        }

    def TrainClassifier(self):
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
        print(f"Classifier loss: {self.__latestLoss:.4f}. Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")

    def __PrintTestStatus(self, iteration, totalIterations, epochTime):
        estRemainingTime = ((time.time() - epochTime) / self.RefreshUIEachXStep) * (totalIterations - iteration)
        print(f"Accuracy: {(self.__latestAccuracy*100):.2f}% Progress: {((iteration/totalIterations)*100):.2f}%. Est time left: {self.GetDatetimeFromSeconds(estRemainingTime)}    ", end="\r")

    def __SaveCheckpoint(self):
        if os.path.exists(self.CheckpointPath + 'classifier_checkpoint.index'):
            from ProjectTools import HelperFunctions as hf
            hf.DeleteFolderAndAllContents(self.CheckpointPath)

        ckptPath = self.CheckpointPath + 'classifier_checkpoint_' + str(self.CurrentEpoch)

        self.Classifier.save_weights(ckptPath)
        self.__MakeCheckpointRef(ckptPath, self.LatestCheckpointPath)

    def __MakeCheckpointRef(self, ckptPathSrc, ckptPathDest):
        relPath = os.path.relpath(os.path.abspath(ckptPathSrc), os.path.abspath(ckptPathDest))
        os.makedirs(os.path.dirname(ckptPathDest), exist_ok=True)
        with open(ckptPathDest, 'w') as f:
            f.write(f"{ckptPathSrc}")

    def __LogData(self, epoch):
        self.Logger.AppendToCSV([epoch + 1, self.__latestLoss])
        self.SummaryWriter['CLoss'].LogNumber(self.__latestLoss, epoch)
        self.SummaryWriter['Accuracy'].LogNumber(self.__latestAccuracy, epoch)

    def __EpochRun(self, epoch):
        print("Training Classifier...")
        (image_batch_train, image_batch_test) = self.CreateDataSet(self.Datasets)
        totalIterations = tf.data.experimental.cardinality(image_batch_train).numpy()
        iteration = 0
        epochTime = time.time()
        for image_batch in image_batch_train:
            if iteration % self.RefreshUIEachXStep == 0:
                returnVal = self.Classifier.train_step(image_batch, True)
                self.__latestLoss = float(returnVal['c_loss'])
                self.__PrintStatus(iteration, totalIterations, epochTime, epoch)
                epochTime = time.time()
            else:
                self.Classifier.train_step(image_batch, False)
            iteration += 1
        
        keras.backend.clear_session()

        self.__PrintStatus(totalIterations, totalIterations, epochTime, epoch)
        print("")
        print("Done!")

        print("Testing Classifier...")
        iteration = 0
        for image_batch in image_batch_test:
            if iteration % self.RefreshUIEachXStep == 0:
                returnTest = self.Classifier.test_step(image_batch, True)
                self.__latestAccuracy = float(returnTest['classifier_accuracy'])
                self.__PrintTestStatus(iteration, totalIterations, epochTime)
                epochTime = time.time()
            else:
                self.Classifier.test_step(image_batch, False)
            iteration += 1
        self.Classifier.accuracy.reset_state()

        self.__PrintTestStatus(totalIterations, totalIterations, epochTime)
        print("")
        print("Done!")

        if self.SaveCheckpoints:
            self.__SaveCheckpoint()

        self.__LogData(epoch)

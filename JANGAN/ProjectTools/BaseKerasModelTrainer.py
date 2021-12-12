from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")

import tensorflow as tf
from tensorflow import keras
import time
import os

class BaseKerasModelTrainer():
    Model = None
    Datasets = []
    Epochs = 0
    CurrentEpoch = None
    RefreshUIEachXStep = 1
    SaveCheckpoints = False
    CheckpointPath = ""
    LatestCheckpointPath = ""

    def __init__(self, model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath):
        self.Model = model
        self.Datasets = datasets
        self.Epochs = epochs
        self.RefreshUIEachXStep = refreshUIEachXStep
        self.SaveCheckpoints = saveCheckPoints
        self.CheckpointPath = checkpointPath
        self.LatestCheckpointPath = latestCheckpointPath

    def TrainModel(self):
        print("Training started")
        for epoch in range(self.Epochs):
            self.CurrentEpoch = epoch
            start = time.time()

            print(f"Epoch {epoch + 1} of {self.Epochs} is in progress...")
            
            self.EpochRun(epoch)

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

    def PrintStatus(self, iteration, totalIterations, epochTime, epoch):
        raise Exception("No Train loss text implemented.")

    def PrintTestStatus(self, iteration, totalIterations, epochTime):
        raise Exception("No Test accuracy text implemented.")

    def SaveCheckpoint(self):
        if os.path.exists(self.CheckpointPath + 'checkpoint.index'):
            from ProjectTools import HelperFunctions as hf
            hf.DeleteFolderAndAllContents(self.CheckpointPath)

        ckptPath = self.CheckpointPath + 'checkpoint_' + str(self.CurrentEpoch)

        self.Model.save_weights(ckptPath)
        self.MakeCheckpointRef(ckptPath, self.LatestCheckpointPath)

    def MakeCheckpointRef(self, ckptPathSrc, ckptPathDest):
        relPath = os.path.relpath(os.path.abspath(ckptPathSrc), os.path.abspath(ckptPathDest))
        os.makedirs(os.path.dirname(ckptPathDest), exist_ok=True)
        with open(ckptPathDest, 'w') as f:
            f.write(f"{ckptPathSrc}")

    def LogData(self, epoch):
        raise Exception("No Data logging implemented")

    def EpochRun(self, epoch):
        (image_batch_train, image_batch_test) = self.CreateDataSet(self.Datasets)
        totalTrainIterations = self.GetDatasetSize(image_batch_train)
        totalTestIterations = self.GetDatasetSize(image_batch_test)

        if totalTrainIterations > 0:
            print("Training Model...")
            iteration = 0
            epochTime = time.time()
            for image_batch in image_batch_train:
                if iteration % self.RefreshUIEachXStep == 0:
                    returnVal = self.Model.train_step(image_batch, True)
                    self.SetTrainProperties(returnVal)
                    self.PrintStatus(iteration, totalTrainIterations, epochTime, epoch)
                    epochTime = time.time()
                else:
                    self.Model.train_step(image_batch, False)
                iteration += 1
        
            keras.backend.clear_session()

            self.PrintStatus(totalTrainIterations, totalTrainIterations, epochTime, epoch)
            print("")
            print("Done!")

        if totalTestIterations > 0:
            print("Testing Model...")
            iteration = 0
            for image_batch in image_batch_test:
                if iteration % self.RefreshUIEachXStep == 0:
                    returnTest = self.Model.test_step(image_batch, True)
                    self.SetTestProperties(returnTest)
                    self.PrintTestStatus(iteration, totalTestIterations, epochTime)
                    epochTime = time.time()
                else:
                    self.Model.test_step(image_batch, False)
                iteration += 1

            self.PrintTestStatus(totalTestIterations, totalTestIterations, epochTime)
            print("")
            print("Done!")

        if self.SaveCheckpoints:
            self.SaveCheckpoint()

        self.LogData(epoch)

    def GetDatasetSize(self, data):
        return tf.data.experimental.cardinality(data).numpy()

    def SetTrainProperties(self, returnVal):
        pass

    def SetTestProperties(self, returnTest):
        pass
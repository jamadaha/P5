from ProjectTools import CSVLogger
from ProjectTools import TFLogger
from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import BaseMLModel as bm

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("tqdm")
ap.CheckAndInstall("numpy")

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm

from DatasetLoader import DatasetLoader as dl
from DatasetLoader import DatasetFormatter as df
from Classifier import ClassifierKerasModel as cm
from Classifier import LayerDefinition as ld
from Classifier import ClassifierTrainer as ct

class ClassifierMLModel(bm.BaseMLModel):
    ClassifyDir = ""
    FormatClassificationImages = False

    LearningRateClass = 0

    PredictionCount = {}
    CorrectPredictions = {}
    IncorrectPredictions = {}

    Classifier = None
    Logger = None
    SummaryWriter = None

    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, epochCount, refreshEachStep, trainingDataDir, classifyDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateClass, formatImages, formatClassificationImages):
        super().__init__(batchSize, numberOfChannels, numberOfClasses, imageSize, None, epochCount, refreshEachStep, trainingDataDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, formatImages)
        self.ClassifyDir = classifyDir
        self.LearningRateClass = learningRateClass
        self.FormatClassificationImages = formatClassificationImages
        self.Logger = CSVLogger.CSVLogger(logPath, 'TestData')
        self.Logger.InitCSV(['Index', 'Correct', 'Inccorect'])
        self.SummaryWriter = {
            'ConfMatrix': TFLogger.TFLogger(logPath, 'ConfMatrix', 'CPredictions')
        }

    def SetupModel(self):
        layerDefiniton = ld.LayerDefinition(self.NumberOfClasses, self.ImageSize, self.NumberOfChannels)

        self.Classifier = cm.ClassifierModel(
            classifier=layerDefiniton.GetClassifier(), 
            imageSize=self.ImageSize, 
            numberOfClasses=self.NumberOfClasses
        )

        self.__Compile()

        self.Trainer = ct.ClassifierTrainer(self.Classifier, self.TensorDatasets, self.EpochCount, self.RefreshEachStep, self.SaveCheckpoints, self.CheckpointPath, self.LatestCheckpointPath, self.LogPath)

    def __Compile(self):
        optimizer = self.__GetOptimizer()
        lossFunc = self.__GetLossFunction()

        self.Classifier.compile(
            optimizer=optimizer,
            loss_fn=lossFunc
        )
    
    def __GetOptimizer(self):
        learningSchedule = self.__GetLearningSchedule()
        return (
            keras.optimizers.Adam(learning_rate=learningSchedule)
        )

    def __GetLearningSchedule(self):
        if self.LRScheduler == 'Constant':
            return self.LearningRateClass
        elif self.LRScheduler == 'ExponentialDecay':
            return (
                keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.LearningRateClass,
                decay_steps=10000,
                decay_rate=0.9
                )  
            )
    
    def __GetLossFunction(self): 
        return keras.losses.CategoricalCrossentropy(from_logits=True)
    
    def ProduceOutput(self):
        self.UseSavedModel = True
        if self.Classifier == None:
            self.TrainModel()

        dataLoader = dl.DatasetLoader(
            self.ClassifyDir,
            (self.ImageSize,self.ImageSize),
            self.FormatClassificationImages)
        dataLoader.DataSets = []
        dataLoader.LoadDatasets()
        dataArray = dataLoader.DataSets

        predictionArray = []
        labelArray = []

        print("Predicting dataset...")

        totalCorrectPredictions = 0
        totalIncorrectPredictions = 0
        totalPredictionsCount = 0
        for data in tqdm(iterable=dataArray, total=len(dataArray)):
            (images, labels) = data
            datasetFormatter = df.DatasetFormatter(images, labels, self.NumberOfClasses, self.BatchSize, 1)
            classifyData = datasetFormatter.ProcessData()

            correctPredictions = 0
            incorrectPredictions = 0
            currentClass = 0
            for (images, labels) in classifyData:
                currentClass = np.argmax(labels[0])
                if not str(currentClass) in self.CorrectPredictions:
                    self.CorrectPredictions[str(currentClass)] = 0
                if not str(currentClass) in self.IncorrectPredictions:
                    self.IncorrectPredictions[str(currentClass)] = 0
                if not str(currentClass) in self.PredictionCount:
                    self.PredictionCount[str(currentClass)] = 0

                predictions = self.Classifier.classifier(images, training=False)
                for prediction in predictions:
                    predictedClass = np.argmax(prediction)
                    predictionArray.append(predictedClass)
                    labelArray.append(currentClass)

                    if predictedClass == currentClass:
                        self.CorrectPredictions[str(currentClass)] = int(self.CorrectPredictions[str(currentClass)]) + 1
                        correctPredictions += 1
                        totalCorrectPredictions += 1
                    else:
                        self.IncorrectPredictions[str(currentClass)] = int(self.IncorrectPredictions[str(currentClass)]) + 1
                        incorrectPredictions += 1
                        totalIncorrectPredictions += 1

                    self.PredictionCount[str(currentClass)] = int(self.PredictionCount[str(currentClass)]) + 1
                    totalPredictionsCount += 1
                

                
        print("Prediction complete!")

        for key in self.CorrectPredictions:
            print(f"[Index {key}] Classifier predicted: {self.CorrectPredictions[str(key)]} correct, {self.IncorrectPredictions[str(key)]} incorrect, {((self.CorrectPredictions[str(key)]/self.PredictionCount[str(key)])*100):.2f}%")
            self.__LogData(key, self.CorrectPredictions[key], self.IncorrectPredictions[key])

        print(f"Total accuracy of classified dataset: {totalCorrectPredictions} correct, {totalIncorrectPredictions} incorrect, {((totalCorrectPredictions/totalPredictionsCount)*100):.2f}%")

        confMatrix = tf.math.confusion_matrix(labelArray, predictionArray, self.NumberOfClasses)
        self.SummaryWriter["ConfMatrix"].LogConfusionMatrix(confMatrix, 0, True)

    def __LogData(self, index, correct, incorrect):
        self.Logger.AppendToCSV([index, correct, incorrect])
from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("tqdm")

from tensorflow import keras
import os

from DatasetLoader import DatasetLoader as dl
from DatasetLoader import DatasetFormatter as df
from CGAN import CGANKerasModel as km
from CGAN import LayerDefinition as ld
from CGAN import CGANTrainer as ct

class CGAN():
    BatchSize = -1
    NumberOfChannels = -1
    NumberOfClasses = -1
    ImageSize = -1
    LatentDimension = -1
    EpochCount = -1
    RefreshEachStep = -1
    ImageCountToProduce = -1
    TensorDatasets = None
    SaveCheckpoints = True
    UseSavedModel = False
    CheckpointPath = ""
    LatestCheckpointPath = ""
    LogPath = ""

    TrainingDataDir = ""
    TestingDataDir = ""
    DatasetSplit = 0

    #AccuracyThreshold = 0

    LRScheduler = ''
    LearningRateDis = 0.0
    LearningRateGen = 0.0

    CondGAN = None
    DataLoader = None
    TrainedGenerator = None

    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, imageCountToProduce, trainingDataDir, testingDataDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateDis, learningRateGen):
        self.BatchSize = batchSize
        self.NumberOfChannels = numberOfChannels
        self.NumberOfClasses = numberOfClasses
        self.ImageSize = imageSize
        self.LatentDimension = latentDimension
        self.EpochCount = epochCount
        self.RefreshEachStep = refreshEachStep
        self.ImageCountToProduce = imageCountToProduce
        self.TrainingDataDir = trainingDataDir
        self.TestingDataDir = testingDataDir
        self.OutputDir = outputDir
        self.SaveCheckpoints = saveCheckpoints
        self.UseSavedModel = useSavedModel
        self.CheckpointPath = checkpointPath
        self.LatestCheckpointPath = latestCheckpointPath
        self.LogPath = logPath
        self.DatasetSplit = datasetSplit
        self.LRScheduler = LRScheduler
        self.LearningRateDis = learningRateDis
        self.LearningRateGen = learningRateGen

    def SetupCGAN(self):
        generator_in_channels = self.LatentDimension + self.NumberOfClasses
        discriminator_in_channels = self.NumberOfChannels + self.NumberOfClasses

        layerDefiniton = ld.LayerDefinition(discriminator_in_channels,generator_in_channels)

        self.CondGAN = km.ConditionalGAN(
            discriminator=layerDefiniton.GetDiscriminator(), 
            generator=layerDefiniton.GetGenerator(), 
            latentDimension=self.LatentDimension, 
            imageSize=self.ImageSize, 
            numberOfClasses=self.NumberOfClasses,
            #accuracyThreshold=self.AccuracyThreshold
        )

        if self.LRScheduler == 'Constant':
            self.CondGAN.compile(
                d_optimizer=keras.optimizers.Adam(learning_rate=self.LearningRateDis),
                g_optimizer=keras.optimizers.Adam(learning_rate=self.LearningRateGen),
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            )  
        elif self.LRScheduler == 'ExponentialDecay':
            disSchedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.LearningRateDis,
                decay_steps=10000,
                decay_rate=0.9
            )
            genSchedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.LearningRateGen,
                decay_steps=10000,
                decay_rate=0.9
            )

            self.CondGAN.compile(
                d_optimizer=keras.optimizers.Adam(learning_rate=disSchedule),
                g_optimizer=keras.optimizers.Adam(learning_rate=genSchedule),
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            )

    def LoadDataset(self):
        if self.UseSavedModel:
            print("Assuming checkpoint exists. Continuing without loading data...")
            return

        dataLoader = dl.DatasetLoader(
            self.TrainingDataDir,
            self.TestingDataDir,
            (self.ImageSize,self.ImageSize))
        dataLoader.LoadTrainDatasets()
        dataArray = dataLoader.DataSets

        bulkDatasetFormatter = df.BulkDatasetFormatter(dataArray, self.NumberOfClasses,self.BatchSize, self.DatasetSplit)
        self.TensorDatasets = bulkDatasetFormatter.ProcessData()

    def TrainGAN(self):
        checkpointPath = self.__GetCheckpointPath()
        if not checkpointPath:
            print("Checkpoint not found! Training instead")
            self.UseSavedModel = False
            if self.TensorDatasets == None:
                self.LoadDataset()

        cGANTrainer = ct.CGANTrainer(self.CondGAN, self.TensorDatasets, self.EpochCount, self.RefreshEachStep, self.SaveCheckpoints, self.CheckpointPath, self.LatestCheckpointPath, self.LogPath, self.OutputDir, self.ImageCountToProduce, self.NumberOfClasses, self.LatentDimension)

        if self.UseSavedModel:
            print("Attempting to load CGAN model from checkpoint...")
            cGANTrainer.CGAN.load_weights(checkpointPath).expect_partial()
            print("Checkpoint loaded!")
        else:
            cGANTrainer.TrainCGAN()
        self.TrainedGenerator = cGANTrainer.CGAN.generator

    def __GetCheckpointPath(self):
        if not os.path.exists(self.LatestCheckpointPath):
            return None

        with open(self.LatestCheckpointPath, 'r') as f:
            ckptPath = f.readline().strip()

        if not os.path.exists(f"{ckptPath}.index"):
            return None
        else:
            return ckptPath

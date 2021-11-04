from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("keras")

from tensorflow import keras
import os

from CGAN import DatasetLoader as dl
from CGAN import DatasetFormatter as df
from CGAN import CGANKerasModel as km
from CGAN import LayerDefinition as ld
from CGAN import LetterProducer as lp
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

    TrainingDataDir = ""
    TestingDataDir = ""

    CondGAN = None
    DataLoader = None
    TrainedGenerator = None

    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, imageCountToProduce, trainingDataDir, testingDataDir, saveCheckpoints, useSavedModel, checkpointPath):
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
        self.SaveCheckpoints = saveCheckpoints
        self.UseSavedModel = useSavedModel
        self.CheckpointPath = checkpointPath

    def SetupCGAN(self):
        generator_in_channels = self.LatentDimension + self.NumberOfClasses
        discriminator_in_channels = self.NumberOfChannels + self.NumberOfClasses

        layerDefiniton = ld.LayerDefinition(discriminator_in_channels,generator_in_channels)

        self.CondGAN = km.ConditionalGAN(
            discriminator=layerDefiniton.GetDiscriminator(), 
            generator=layerDefiniton.GetGenerator(), 
            latentDimension=self.LatentDimension, 
            imageSize=self.ImageSize, 
            numberOfClasses=self.NumberOfClasses
        )
        self.CondGAN.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
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

        bulkDatasetFormatter = df.BulkDatasetFormatter(dataArray, self.NumberOfClasses,self.BatchSize)
        self.TensorDatasets = bulkDatasetFormatter.ProcessData();

    def TrainGAN(self):
        if not os.path.exists(self.CheckpointPath + 'cgan_checkpoint.index'):
            print("Checkpoint not found! Training instead")
            self.UseSavedModel = False
            self.LoadDataset()

        cGANTrainer = ct.CGANTrainer(self.CondGAN,self.TensorDatasets,self.EpochCount,self.RefreshEachStep,self.SaveCheckpoints, self.CheckpointPath)

        if self.UseSavedModel:
            print("Attempting to load CGAN model from checkpoint...")
            cGANTrainer.CGAN.load_weights(self.CheckpointPath + 'cgan_checkpoint')
            print("Checkpoint loaded!")
        else:
            cGANTrainer.TrainCGAN()
        self.TrainedGenerator = cGANTrainer.CGAN.generator

    def ProduceLetters(self):
        sentinel = True
        while(sentinel):
            Question = input(f"Enter a new index to generate (0-{self.NumberOfClasses - 1}))(type N to exit):")
            if Question == "N":
                sentinel = False
                break

            if not Question.isnumeric():
                print("Please only write numbers or N to exit")
                continue

            value = int(Question)

            if value >= self.NumberOfClasses:
                print(f"Please write numbers within 0-{self.NumberOfClasses - 1}")
                continue
            if value < 0:
                print(f"Please write numbers within 0-{self.NumberOfClasses - 1}")
                continue

            letterProducer = lp.LetterProducer(self.TrainedGenerator, self.NumberOfClasses, self.LatentDimension)

            images = letterProducer.GenerateLetter(value, self.ImageCountToProduce)
            letterProducer.SaveImagesAsGif(images)
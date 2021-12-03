from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import BaseMLModel as bm

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("tqdm")

from tensorflow import keras
import os
from tqdm import tqdm

from CGAN import CGANKerasModel as km
from CGAN import LayerDefinition as ld
from CGAN import CGANTrainer as ct
from CGAN import LetterProducer

class CGANMLModel(bm.BaseMLModel):
    ImageCountToProduce = -1

    LearningRateDis = 0.0
    LearningRateGen = 0.0

    EpochImgDir = ""

    TrainedGenerator = None

    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, imageCountToProduce, trainingDataDir, testingDataDir, outputDir, epochImgDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateDis, learningRateGen, formatImages):
        super().__init__(batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, trainingDataDir, testingDataDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, formatImages)
        self.ImageCountToProduce = imageCountToProduce
        self.LearningRateDis = learningRateDis
        self.LearningRateGen = learningRateGen
        self.EpochImgDir = epochImgDir

    def SetupModel(self):
        generator_in_channels = self.LatentDimension + self.NumberOfClasses
        discriminator_in_channels = self.NumberOfChannels + self.NumberOfClasses

        layerDefiniton = ld.LayerDefinition(discriminator_in_channels,generator_in_channels)

        self.KerasModel = km.ConditionalGAN(
            discriminator=layerDefiniton.GetDiscriminator(), 
            generator=layerDefiniton.GetGenerator(), 
            latentDimension=self.LatentDimension, 
            imageSize=self.ImageSize, 
            numberOfClasses=self.NumberOfClasses,
        )

        if self.LRScheduler == 'Constant':
            self.KerasModel.compile(
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

            self.KerasModel.compile(
                d_optimizer=keras.optimizers.Adam(learning_rate=disSchedule),
                g_optimizer=keras.optimizers.Adam(learning_rate=genSchedule),
                loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
            )

        self.Trainer = ct.CGANTrainer(self.KerasModel, self.TensorDatasets, self.EpochCount, self.RefreshEachStep, self.SaveCheckpoints, self.CheckpointPath, self.LatestCheckpointPath, self.LogPath, self.NumberOfClasses, self.LatentDimension, self.EpochImgDir)

    def TrainModel(self):
        super().TrainModel()
        self.TrainedGenerator = self.Trainer.Model.generator

    def ProduceOutput(self):
        self.UseSavedModel = True
        if self.TrainedGenerator == None:
            self.TrainModel()

        letterProducer = LetterProducer.LetterProducer(self.OutputDir, self.TrainedGenerator, self.NumberOfClasses, self.LatentDimension, self.ImageCountToProduce)
        letterProducer.ProduceLetters()
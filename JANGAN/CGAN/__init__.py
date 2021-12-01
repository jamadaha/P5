from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import BaseMLModel as bm

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("tqdm")

from tensorflow import keras
import os
from tqdm import tqdm

from CGAN import CGANKerasModel as km
from CGAN import LayerDefinition as ld
from CGAN import LetterProducer as lp
from CGAN import CGANTrainer as ct

class CGAN(bm.BaseMLModel):
    ImageCountToProduce = -1

    LearningRateDis = 0.0
    LearningRateGen = 0.0

    TrainedGenerator = None

    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, imageCountToProduce, trainingDataDir, testingDataDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateDis, learningRateGen):
        super().__init__(batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, trainingDataDir, testingDataDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler)
        self.ImageCountToProduce = imageCountToProduce
        self.LearningRateDis = learningRateDis
        self.LearningRateGen = learningRateGen

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

        self.Trainer = ct.CGANTrainer(self.KerasModel, self.TensorDatasets, self.EpochCount, self.RefreshEachStep, self.SaveCheckpoints, self.CheckpointPath, self.LatestCheckpointPath, self.LogPath)

    def TrainModel(self):
        super().TrainModel()
        self.TrainedGenerator = self.Trainer.Model.generator

    def ProduceOutput(self):
        self.UseSavedModel = True
        if self.TrainedGenerator == None:
            self.TrainGAN()

        letterProducer = lp.LetterProducer(self.OutputDir, self.TrainedGenerator, self.NumberOfClasses, self.LatentDimension)
        # Warmup letter producer
        #   This is done as it outputs something to console
        letterProducer.GenerateLetter(0, 1)

        for i in tqdm(range(self.NumberOfClasses), desc='Producing images'):
            images = letterProducer.GenerateLetter(i, self.ImageCountToProduce)
            letterProducer.SaveImages(i, images)
           
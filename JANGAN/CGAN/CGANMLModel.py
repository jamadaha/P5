from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import BaseMLModel as bm

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("tqdm")

from tensorflow import keras

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

    TrackModeCollapse = False
    ModeCollapseThreshold = 0

    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, imageCountToProduce, trainingDataDir, outputDir, epochImgDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateDis, learningRateGen, formatImages, trackModeCollapse, modeCollapseThreshold):
        super().__init__(batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, trainingDataDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, formatImages)
        self.ImageCountToProduce = imageCountToProduce
        self.LearningRateDis = learningRateDis
        self.LearningRateGen = learningRateGen
        self.EpochImgDir = epochImgDir
        self.TrackModeCollapse = trackModeCollapse
        self.ModeCollapseThreshold = modeCollapseThreshold

    def SetupModel(self):
        generator_in_channels = self.LatentDimension + self.NumberOfClasses
        discriminator_in_channels = self.NumberOfChannels + self.NumberOfClasses

        layerDefiniton = ld.LayerDefinition(self.ImageSize, discriminator_in_channels, generator_in_channels)

        self.KerasModel = km.CGANKerasModel(
            layerDefiniton.GetDiscriminator(), 
            layerDefiniton.GetGenerator(), 
            self.LatentDimension, 
            self.ImageSize, 
            self.NumberOfClasses,
            self.TrackModeCollapse
        )

        self.Compile()

        self.Trainer = ct.CGANTrainer(self.KerasModel, self.TensorDatasets, self.EpochCount, self.RefreshEachStep, self.SaveCheckpoints, self.CheckpointPath, self.LatestCheckpointPath, self.LogPath, self.NumberOfClasses, self.LatentDimension, self.EpochImgDir, self.TrackModeCollapse, self.ModeCollapseThreshold)

    def Compile(self):
        (disOptimizer, genOptimizer) = self.GetOptimizer()
        self.KerasModel.compile(
                disOptimizer,
                genOptimizer,
                self.GetLossFunction()
        )

    def GetOptimizer(self):
        (disSchedule, genSchedule) = self.GetLearningSchedule()
        return (
            keras.optimizers.Adam(learning_rate=disSchedule),
            keras.optimizers.Adam(learning_rate=genSchedule)
        )

    def GetLearningSchedule(self):
        if self.LRScheduler == 'Constant':
            return (self.LearningRateDis, self.LearningRateGen) 
        elif self.LRScheduler == 'ExponentialDecay':
            return (
                keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.LearningRateDis,
                decay_steps=10000,
                decay_rate=0.9
                ),
                keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.LearningRateGen,
                decay_steps=10000,
                decay_rate=0.9
                )   
            )

    def GetLossFunction(self): 
        return keras.losses.BinaryCrossentropy(from_logits=True)
        
    def TrainModel(self):
        super().TrainModel()
        self.TrainedGenerator = self.Trainer.Model.Generator

    def ProduceOutput(self):
        self.UseSavedModel = True
        if self.TrainedGenerator == None:
            self.TrainModel()

        letterProducer = LetterProducer.LetterProducer(self.OutputDir, self.TrainedGenerator, self.NumberOfClasses, self.LatentDimension, self.ImageCountToProduce)
        letterProducer.ProduceLetters()

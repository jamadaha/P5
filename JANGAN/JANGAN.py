import importlib
import os

from importlib import reload

from ProjectTools import ConfigHelper
 
import CGAN as cg
import DataGenerator as dg
import Classifier as cf

class JANGAN():
    cfg = None
    cgan = None
    classifier = None
    NumerOfClasses = None

    def __init__(self, expFile, configFile):
        importlib.import_module(expFile)
        self.LoadConfig(configFile)

    def LoadConfig(self, configFile):
        print(" --- Loading experiment config file --- ")
        self.cfg = ConfigHelper.ConfigHelper(configFile)
        self.cfg.LoadConfig()
        print(" --- Done! --- ")
        print("")

    def PurgeRunDataFolder(self):
        print(" --- Purging training data folder --- ")

        from ProjectTools import HelperFunctions as hf
        hf.DeleteFolderAndAllContents(self.cfg.GetStringValue("DATAGENERATOR","LetterPath"))

        print(f" --- Done! --- ")

    def MakeCGANDataset(self):
        if self.cfg.GetBoolValue("DATAGENERATOR", "PurgePreviousData"):
            self.PurgeRunDataFolder()

            
        self.cfg.CopyConfigToPath(self.cfg.GetStringValue("CGAN", "ConfigCopyPath"))

        print(" --- Generating dataset if not there --- ")

        datagen = dg.DataGenerator(
            self.cfg.GetStringValue("DATAGENERATOR", "LetterDownloadURL"),
            self.cfg.GetStringValue("DATAGENERATOR", "LetterDownloadPath"),
            self.cfg.GetStringValue("DATAGENERATOR", "LetterDownloadName"),
            self.cfg.GetStringValue("DATAGENERATOR", "LetterPath"),
            self.cfg.GetStringValue("DATAGENERATOR", "LetterOutputFormat"),
            self.cfg.GetIntValue("DATAGENERATOR", "MinimumLetterCount"),
            self.cfg.GetIntValue("DATAGENERATOR", "MaximumLetterCount"),
            self.cfg.GetJsonValue("DATAGENERATOR", "TextDownloadURLS"),
            self.cfg.GetStringValue("DATAGENERATOR", "TextPath"),
            self.cfg.GetStringValue("DATAGENERATOR", "DistributionPath"),
            self.cfg.GetBoolValue("DATAGENERATOR", "PrintDistribution"),
            self.cfg.GetBoolValue("DATAGENERATOR", "IncludeNumbers"),
            self.cfg.GetBoolValue("DATAGENERATOR", "IncludeLetters"),
        )
        datagen.GenerateData()

        print(" --- Done! --- ")
        print("")

    def TrainCGAN(self):
        print(" --- Training CGAN --- ")

        self.NumerOfClasses = 0
        for entry in os.scandir(self.cfg.GetStringValue("DATAGENERATOR", "LetterPath")):
            if entry.is_dir():
                self.NumerOfClasses += 1

        self.cgan = cg.CGAN(
            self.cfg.GetIntValue("CGAN", "BatchSize"),
            1,
            self.NumerOfClasses,
            self.cfg.GetIntValue("CGAN", "ImageSize"),
            self.cfg.GetIntValue("CGAN", "LatentDimension"),
            self.cfg.GetIntValue("CGAN", "EpochCount"),
            self.cfg.GetIntValue("CGAN", "RefreshUIEachXIteration"),
            self.cfg.GetIntValue("CGAN", "NumberOfFakeImagesToOutput"),
            self.cfg.GetStringValue("CGAN", "TrainDatasetDir"),
            self.cfg.GetStringValue("CGAN", "TestDatasetDir"),
            self.cfg.GetStringValue("CGAN", "OutputDir"),
            self.cfg.GetBoolValue("CGAN", "SaveCheckpoints"),
            self.cfg.GetBoolValue("CGAN", "UseSavedModel"),
            self.cfg.GetStringValue("CGAN", "CheckpointPath"),
            self.cfg.GetStringValue("CGAN", "LatestCheckpointPath"),
            self.cfg.GetStringValue("CGAN", "LogPath"),
            self.cfg.GetFloatValue("CGAN", "DatasetSplit"),
            self.cfg.GetStringValue("CGAN", "LRScheduler"),
            self.cfg.GetFloatValue("CGAN", "LearningRateDiscriminator"),
            self.cfg.GetFloatValue("CGAN", "LearningRateGenerator"))

        self.cgan.SetupCGAN()
        self.cgan.LoadDataset()
        self.cgan.TrainGAN()

        print(" --- Done! --- ")

    def ProduceOutput(self):
        print(" --- Producing output --- ")

        self.cgan.ProduceLetters()

        print(" --- Done! --- ")

    def ClassifyCGANOutput(self):
        print(" --- Classifying Output of CGAN --- ")

        if self.NumerOfClasses == None:
            self.NumerOfClasses = 0
            for entry in os.scandir(self.cfg.GetStringValue("Classifier", "TrainDatasetDir")):
                if entry.is_dir():
                    self.NumerOfClasses += 1

        self.classifier = cf.Classifier(
            self.cfg.GetIntValue("Classifier", "BatchSize"),
            1,
            self.NumerOfClasses,
            self.cfg.GetIntValue("Classifier", "ImageSize"),
            self.cfg.GetIntValue("Classifier", "EpochCount"),
            self.cfg.GetIntValue("Classifier", "RefreshUIEachXIteration"),
            self.cfg.GetStringValue("Classifier", "TrainDatasetDir"),
            self.cfg.GetStringValue("Classifier", "TestDatasetDir"),
            self.cfg.GetStringValue("Classifier", "ClassifyDir"),
            self.cfg.GetBoolValue("Classifier", "SaveCheckpoints"),
            self.cfg.GetBoolValue("Classifier", "UseSavedModel"),
            self.cfg.GetStringValue("Classifier", "CheckpointPath"),
            self.cfg.GetStringValue("Classifier", "LatestCheckpointPath"),
            self.cfg.GetStringValue("Classifier", "LogPath"),
            self.cfg.GetFloatValue("Classifier", "DatasetSplit"),
            self.cfg.GetStringValue("Classifier", "LRScheduler"),
            self.cfg.GetFloatValue("Classifier", "LearningRateClassifier"),
            self.cfg.GetFloatValue("Classifier", "AccuracyThreshold"))

        self.classifier.SetupClassifier()
        self.classifier.LoadDataset()
        self.classifier.TrainClassifier()

        self.classifier.ClassifyData()

        print(" --- Done! --- ")
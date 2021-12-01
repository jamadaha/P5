import importlib
import os
from JANGANConfigChecker import JANGANConfigChecker

from ProjectTools import ConfigHelper
 
import CGAN as cg
import DataGenerator as dg
import Classifier as cf


class JANGAN():
    cfg = None
    cgan = None
    classifier = None
    NumberOfClasses = None
    ThrowIfConfigFileBad = True

    def __init__(self, expFile, configFile, throwIfConfigFileBad):
        importlib.import_module(expFile)
        self.ThrowIfConfigFileBad = throwIfConfigFileBad
        self.LoadConfig(configFile)

    def LoadConfig(self, configFile):
        print(" --- Loading experiment config file --- ")
        self.cfg = ConfigHelper.ConfigHelper(configFile)
        self.cfg.LoadConfig()
        print(" --- Done! --- ")
        cfgChecker = JANGANConfigChecker()
        cfgChecker.CheckConfig(self.cfg, self.ThrowIfConfigFileBad)
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

    def __GetNumberOfClasses(self):
        self.NumberOfClasses = 0
        for entry in os.scandir(self.cfg.GetStringValue("DATAGENERATOR", "LetterPath")):
            if entry.is_dir():
                self.NumberOfClasses += 1

    def __SetupCGAN(self):
        self.__GetNumberOfClasses()

        self.cgan = cg.CGAN(
            self.cfg.GetIntValue("CGAN", "BatchSize"),
            1,
            self.NumberOfClasses,
            self.cfg.GetIntValue("CGAN", "ImageSize"),
            self.cfg.GetIntValue("CGAN", "LatentDimension"),
            self.cfg.GetIntValue("CGAN", "EpochCount"),
            self.cfg.GetIntValue("CGAN", "RefreshUIEachXIteration"),
            self.cfg.GetIntValue("CGANOutput", "NumberOfFakeImagesToOutput"),
            self.cfg.GetStringValue("CGAN", "TrainDatasetDir"),
            self.cfg.GetStringValue("CGAN", "TestDatasetDir"),
            self.cfg.GetStringValue("CGANOutput", "OutputDir"),
            self.cfg.GetBoolValue("CGAN", "SaveCheckpoints"),
            self.cfg.GetBoolValue("CGAN", "UseSavedModel"),
            self.cfg.GetStringValue("CGAN", "CheckpointPath"),
            self.cfg.GetStringValue("CGAN", "LatestCheckpointPath"),
            self.cfg.GetStringValue("CGAN", "LogPath"),
            self.cfg.GetFloatValue("CGAN", "DatasetSplit"),
            self.cfg.GetStringValue("CGAN", "LRScheduler"),
            self.cfg.GetFloatValue("CGAN", "LearningRateDiscriminator"),
            self.cfg.GetFloatValue("CGAN", "LearningRateGenerator"))

    def TrainCGAN(self):
        print(" --- Training CGAN --- ")

        if self.cgan == None:
            self.__SetupCGAN()

        self.cgan.SetupCGAN()
        self.cgan.TrainGAN()

        print(" --- Done! --- ")

    def ProduceOutput(self):
        print(" --- Producing output --- ")

        if self.cgan == None:
            self.__SetupCGAN()

        self.cgan.ProduceLetters()

        print(" --- Done! --- ")

    def __SetupClassifier(self):
        self.__GetNumberOfClasses()

        self.classifier = cf.Classifier(
            self.cfg.GetIntValue("Classifier", "BatchSize"),
            1,
            self.NumberOfClasses,
            self.cfg.GetIntValue("Classifier", "ImageSize"),
            self.cfg.GetIntValue("Classifier", "EpochCount"),
            self.cfg.GetIntValue("Classifier", "RefreshUIEachXIteration"),
            self.cfg.GetStringValue("Classifier", "TrainDatasetDir"),
            self.cfg.GetStringValue("Classifier", "TestDatasetDir"),
            self.cfg.GetStringValue("ClassifierOutput", "ClassifyDir"),
            self.cfg.GetBoolValue("Classifier", "SaveCheckpoints"),
            self.cfg.GetBoolValue("Classifier", "UseSavedModel"),
            self.cfg.GetStringValue("Classifier", "CheckpointPath"),
            self.cfg.GetStringValue("Classifier", "LatestCheckpointPath"),
            self.cfg.GetStringValue("Classifier", "LogPath"),
            self.cfg.GetFloatValue("Classifier", "DatasetSplit"),
            self.cfg.GetStringValue("Classifier", "LRScheduler"),
            self.cfg.GetFloatValue("Classifier", "LearningRateClassifier"),
            self.cfg.GetFloatValue("Classifier", "AccuracyThreshold"))

    def TrainClassifier(self):
        print(" --- Training Classifier --- ")

        if self.classifier == None:
            self.__SetupClassifier()

        self.classifier.SetupClassifier()
        self.classifier.TrainClassifier()

        print(" --- Done! --- ")

    def ClassifyCGANOutput(self):
        print(" --- Classifying Output of CGAN --- ")

        if self.classifier == None:
            self.__SetupClassifier()

        self.classifier.ClassifyData()

        print(" --- Done! --- ")
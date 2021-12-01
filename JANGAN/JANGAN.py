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
        self.cfg.CopyConfigToPath(self.cfg.GetStringValue("GLOBAL", "ConfigCopyPath"))
        print("")

    def PurgeRunDataFolder(self, path):
        print(" --- Purging training data folder --- ")

        from ProjectTools import HelperFunctions as hf
        hf.DeleteFolderAndAllContents(path)

        print(f" --- Done! --- ")

    def MakeCGANDataset(self):
        if self.cfg.GetBoolValue("CGANDATAGENERATOR", "PurgePreviousData"):
            self.PurgeRunDataFolder(self.cfg.GetStringValue("CGANDATAGENERATOR","LetterPath"))
            
        print(" --- Generating dataset for CGAN if not there --- ")

        datagen = dg.DataGenerator(
            self.cfg.GetStringValue("CGANDATAGENERATOR", "LetterDownloadURL"),
            self.cfg.GetStringValue("CGANDATAGENERATOR", "LetterDownloadPath"),
            self.cfg.GetStringValue("CGANDATAGENERATOR", "LetterDownloadName"),
            self.cfg.GetStringValue("CGANDATAGENERATOR", "LetterPath"),
            self.cfg.GetStringValue("CGANDATAGENERATOR", "LetterOutputFormat"),
            self.cfg.GetIntValue("CGANDATAGENERATOR", "MinimumLetterCount"),
            self.cfg.GetIntValue("CGANDATAGENERATOR", "MaximumLetterCount"),
            self.cfg.GetJsonValue("CGANDATAGENERATOR", "TextDownloadURLS"),
            self.cfg.GetStringValue("CGANDATAGENERATOR", "TextPath"),
            self.cfg.GetStringValue("CGANDATAGENERATOR", "DistributionPath"),
            self.cfg.GetBoolValue("CGANDATAGENERATOR", "PrintDistribution"),
            self.cfg.GetBoolValue("CGANDATAGENERATOR", "IncludeNumbers"),
            self.cfg.GetBoolValue("CGANDATAGENERATOR", "IncludeLetters"),
        )
        datagen.GenerateData()

        print(" --- Done! --- ")
        print("")

    def MakeClassifyerDataset(self):
        if self.cfg.GetBoolValue("ClassifierDATAGENERATOR", "PurgePreviousData"):
            self.PurgeRunDataFolder(self.cfg.GetStringValue("ClassifierDATAGENERATOR","LetterPath"))

        print(" --- Generating dataset for Classifier if not there --- ")

        datagen = dg.DataGenerator(
            self.cfg.GetStringValue("ClassifierDATAGENERATOR", "LetterDownloadURL"),
            self.cfg.GetStringValue("ClassifierDATAGENERATOR", "LetterDownloadPath"),
            self.cfg.GetStringValue("ClassifierDATAGENERATOR", "LetterDownloadName"),
            self.cfg.GetStringValue("ClassifierDATAGENERATOR", "LetterPath"),
            self.cfg.GetStringValue("ClassifierDATAGENERATOR", "LetterOutputFormat"),
            self.cfg.GetIntValue("ClassifierDATAGENERATOR", "MinimumLetterCount"),
            self.cfg.GetIntValue("ClassifierDATAGENERATOR", "MaximumLetterCount"),
            self.cfg.GetJsonValue("ClassifierDATAGENERATOR", "TextDownloadURLS"),
            self.cfg.GetStringValue("ClassifierDATAGENERATOR", "TextPath"),
            self.cfg.GetStringValue("ClassifierDATAGENERATOR", "DistributionPath"),
            self.cfg.GetBoolValue("ClassifierDATAGENERATOR", "PrintDistribution"),
            self.cfg.GetBoolValue("ClassifierDATAGENERATOR", "IncludeNumbers"),
            self.cfg.GetBoolValue("ClassifierDATAGENERATOR", "IncludeLetters"),
        )
        datagen.GenerateData()

        print(" --- Done! --- ")
        print("")

    def __GetNumberOfCGANClasses(self):
        self.NumberOfClasses = 0
        for entry in os.scandir(self.cfg.GetStringValue("CGANDATAGENERATOR", "LetterPath")):
            if entry.is_dir():
                self.NumberOfClasses += 1

    def __GetNumberOfClassifierClasses(self):
        self.NumberOfClasses = 0
        for entry in os.scandir(self.cfg.GetStringValue("ClassifierDATAGENERATOR", "LetterPath")):
            if entry.is_dir():
                self.NumberOfClasses += 1

    def __SetupCGAN(self):
        self.__GetNumberOfCGANClasses()

        self.cgan = cg.CGAN(
            self.cfg.GetIntValue("CGANTraining", "BatchSize"),
            1,
            self.NumberOfClasses,
            self.cfg.GetIntValue("CGANTraining", "ImageSize"),
            self.cfg.GetIntValue("CGANTraining", "LatentDimension"),
            self.cfg.GetIntValue("CGANTraining", "EpochCount"),
            self.cfg.GetIntValue("CGANTraining", "RefreshUIEachXIteration"),
            self.cfg.GetIntValue("CGANOutput", "NumberOfFakeImagesToOutput"),
            self.cfg.GetStringValue("CGANTraining", "TrainDatasetDir"),
            self.cfg.GetStringValue("CGANTraining", "TestDatasetDir"),
            self.cfg.GetStringValue("CGANOutput", "OutputDir"),
            self.cfg.GetBoolValue("CGANTraining", "SaveCheckpoints"),
            self.cfg.GetBoolValue("CGANTraining", "UseSavedModel"),
            self.cfg.GetStringValue("CGANTraining", "CheckpointPath"),
            self.cfg.GetStringValue("CGANTraining", "LatestCheckpointPath"),
            self.cfg.GetStringValue("CGANTraining", "LogPath"),
            self.cfg.GetFloatValue("CGANTraining", "DatasetSplit"),
            self.cfg.GetStringValue("CGANTraining", "LRScheduler"),
            self.cfg.GetFloatValue("CGANTraining", "LearningRateDiscriminator"),
            self.cfg.GetFloatValue("CGANTraining", "LearningRateGenerator"))

    def TrainCGAN(self):
        print(" --- Training CGAN --- ")

        if self.cgan == None:
            self.__SetupCGAN()

        self.cgan.SetupModel()
        self.cgan.TrainModel()

        print(" --- Done! --- ")

    def ProduceOutput(self):
        print(" --- Producing output --- ")

        if self.cgan == None:
            self.__SetupCGAN()

        self.cgan.ProduceOutput()

        print(" --- Done! --- ")

    def __SetupClassifier(self):
        self.__GetNumberOfClassifierClasses()

        self.classifier = cf.Classifier(
            self.cfg.GetIntValue("ClassifierTraining", "BatchSize"),
            1,
            self.NumberOfClasses,
            self.cfg.GetIntValue("ClassifierTraining", "ImageSize"),
            self.cfg.GetIntValue("ClassifierTraining", "EpochCount"),
            self.cfg.GetIntValue("ClassifierTraining", "RefreshUIEachXIteration"),
            self.cfg.GetStringValue("ClassifierTraining", "TrainDatasetDir"),
            self.cfg.GetStringValue("ClassifierTraining", "TestDatasetDir"),
            self.cfg.GetStringValue("ClassifierOutput", "ClassifyDir"),
            self.cfg.GetStringValue("ClassifierOutput", "OutputDir"),
            self.cfg.GetBoolValue("ClassifierTraining", "SaveCheckpoints"),
            self.cfg.GetBoolValue("ClassifierTraining", "UseSavedModel"),
            self.cfg.GetStringValue("ClassifierTraining", "CheckpointPath"),
            self.cfg.GetStringValue("ClassifierTraining", "LatestCheckpointPath"),
            self.cfg.GetStringValue("ClassifierTraining", "LogPath"),
            self.cfg.GetFloatValue("ClassifierTraining", "DatasetSplit"),
            self.cfg.GetStringValue("ClassifierTraining", "LRScheduler"),
            self.cfg.GetFloatValue("ClassifierTraining", "LearningRateClassifier"),
            self.cfg.GetFloatValue("ClassifierTraining", "AccuracyThreshold"))

    def TrainClassifier(self):
        print(" --- Training Classifier --- ")

        if self.classifier == None:
            self.__SetupClassifier()

        self.classifier.SetupModel()
        self.classifier.TrainModel()

        print(" --- Done! --- ")

    def ClassifyCGANOutput(self):
        print(" --- Classifying Output of CGAN --- ")

        if self.classifier == None:
            self.__SetupClassifier()

        self.classifier.ProduceOutput()

        print(" --- Done! --- ")
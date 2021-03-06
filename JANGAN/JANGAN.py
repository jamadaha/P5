from ProjectTools import AutoPackageInstaller as ap
from JANGANConfigChecker import JANGANConfigChecker
from ProjectTools import ConfigHelper as ch
from ProjectTools import HelperFunctions as hf
from CGAN import CGANMLModel as cg
from Classifier import ClassifierMLModel as cf
import DataGenerator as dg

ap.CheckAndInstall("importlib")
ap.CheckAndInstall("os")

import importlib
import os

class JANGAN():
    ExperimentName = ""
    cfg = None
    cgan = None
    classifier = None
    NumberOfClasses = None
    ThrowIfConfigFileBad = True

    def __init__(self, expName, expFile, configFile, throwIfConfigFileBad):
        importlib.import_module(expFile)
        self.ThrowIfConfigFileBad = throwIfConfigFileBad
        self.ExperimentName = expName;
        self.LoadConfig(configFile)

    def LoadConfig(self, configFile):
        print(" --- Loading experiment config file --- ")
        self.cfg = ch.ConfigHelper(configFile)
        self.cfg.LoadConfig()
        newTokens = self.cfg.TokenReplacements.copy()
        newTokens.append(("{EXPERIMENTNAME}", self.ExperimentName))
        newTokens.append(("{BASEPATH}", self.cfg.GetStringValue("GLOBAL", "BasePath")))
        self.cfg.UpdateTokenReplacements(newTokens)
        self.cfg.LoadConfig()
        print(" --- Done! --- ")
        cfgChecker = JANGANConfigChecker(self.cfg, self.ThrowIfConfigFileBad)
        cfgChecker.CheckConfig()
        self.cfg.CopyConfigToPath(self.cfg.GetStringValue("GLOBAL", "ConfigCopyPath"))
        print("")

    def PurgeRunDataFolder(self, path):
        print(" --- Purging training data folder --- ")

        hf.DeleteFolderAndAllContents(path)

        print(f" --- Done! --- ")

    def MakeCGANDataset(self):
        if self.cfg.GetBoolValue("CGANDATAGENERATOR", "PurgePreviousData"):
            self.PurgeRunDataFolder(self.cfg.GetStringValue("CGANDATAGENERATOR","OutputPath"))
            
        print(" --- Generating dataset for CGAN if not there --- ")

        datagen = dg.DataGenerator(
            self.cfg.GetStringValue("DATAGENERATOR", "LetterDownloadURL"),
            self.cfg.GetStringValue("DATAGENERATOR", "LetterDownloadPath"),
            self.cfg.GetStringValue("DATAGENERATOR", "LetterDownloadName"),
            self.cfg.GetStringValue("CGANDATAGENERATOR", "OutputPath"),
            self.cfg.GetStringValue("CGANDATAGENERATOR", "LetterOutputFormat"),
            self.cfg.GetIntValue("CGANDATAGENERATOR", "MinimumLetterCount"),
            self.cfg.GetIntValue("CGANDATAGENERATOR", "MaximumLetterCount"),
            self.cfg.GetJsonValue("DATAGENERATOR", "TextDownloadURLS"),
            self.cfg.GetStringValue("DATAGENERATOR", "TextPath"),
            self.cfg.GetStringValue("CGANDATAGENERATOR", "DistributionPath"),
            self.cfg.GetBoolValue("CGANDATAGENERATOR", "PrintDistribution"),
            self.cfg.GetBoolValue("CGANDATAGENERATOR", "IncludeNumbers"),
            self.cfg.GetBoolValue("CGANDATAGENERATOR", "IncludeLetters"),
        )
        datagen.GenerateData()

        print(" --- Done! --- ")
        print("")

    def MakeClassifyerDataset(self):
        if self.cfg.GetBoolValue("CLASSIFIERDATAGENERATOR", "PurgePreviousData"):
            self.PurgeRunDataFolder(self.cfg.GetStringValue("CLASSIFIERDATAGENERATOR","OutputPath"))

        print(" --- Generating dataset for Classifier if not there --- ")

        datagen = dg.DataGenerator(
            self.cfg.GetStringValue("DATAGENERATOR", "LetterDownloadURL"),
            self.cfg.GetStringValue("DATAGENERATOR", "LetterDownloadPath"),
            self.cfg.GetStringValue("DATAGENERATOR", "LetterDownloadName"),
            self.cfg.GetStringValue("CLASSIFIERDATAGENERATOR", "OutputPath"),
            self.cfg.GetStringValue("CLASSIFIERDATAGENERATOR", "LetterOutputFormat"),
            self.cfg.GetIntValue("CLASSIFIERDATAGENERATOR", "MinimumLetterCount"),
            self.cfg.GetIntValue("CLASSIFIERDATAGENERATOR", "MaximumLetterCount"),
            self.cfg.GetJsonValue("DATAGENERATOR", "TextDownloadURLS"),
            self.cfg.GetStringValue("DATAGENERATOR", "TextPath"),
            self.cfg.GetStringValue("CLASSIFIERDATAGENERATOR", "DistributionPath"),
            self.cfg.GetBoolValue("CLASSIFIERDATAGENERATOR", "PrintDistribution"),
            self.cfg.GetBoolValue("CLASSIFIERDATAGENERATOR", "IncludeNumbers"),
            self.cfg.GetBoolValue("CLASSIFIERDATAGENERATOR", "IncludeLetters"),
        )
        datagen.GenerateData()

        print(" --- Done! --- ")
        print("")

    def __GetNumberOfCGANClasses(self):
        self.NumberOfClasses = 0
        for entry in os.scandir(self.cfg.GetStringValue("CGANDATAGENERATOR", "OutputPath")):
            if entry.is_dir():
                self.NumberOfClasses += 1

    def __GetNumberOfClassifierClasses(self):
        self.NumberOfClasses = 0
        for entry in os.scandir(self.cfg.GetStringValue("CLASSIFIERDATAGENERATOR", "OutputPath")):
            if entry.is_dir():
                self.NumberOfClasses += 1

    def __SetupCGAN(self):
        self.__GetNumberOfCGANClasses()

        self.cgan = cg.CGANMLModel(
            self.cfg.GetIntValue("CGANTRAINING", "BatchSize"),
            self.cfg.GetIntValue("CGANTRAINING", "ImageChannels"),
            self.NumberOfClasses,
            self.cfg.GetIntValue("CGANTRAINING", "ImageSize"),
            self.cfg.GetIntValue("CGANTRAINING", "LatentDimension"),
            self.cfg.GetIntValue("CGANTRAINING", "EpochCount"),
            self.cfg.GetIntValue("CGANTRAINING", "RefreshUIEachXIteration"),
            self.cfg.GetIntValue("CGANOUTPUT", "NumberOfFakeImagesToOutput"),
            self.cfg.GetStringValue("CGANTRAINING", "TrainDatasetDir"),
            self.cfg.GetStringValue("CGANOUTPUT", "OutputDir"),
            self.cfg.GetStringValue("CGANTRAINING", "EpochImgDir"),
            self.cfg.GetBoolValue("CGANTRAINING", "SaveCheckpoints"),
            self.cfg.GetBoolValue("CGANTRAINING", "UseSavedModel"),
            self.cfg.GetStringValue("CGANTRAINING", "CheckpointPath"),
            self.cfg.GetStringValue("CGANTRAINING", "LatestCheckpointPath"),
            self.cfg.GetStringValue("CGANTRAINING", "LogPath"),
            self.cfg.GetFloatValue("CGANTRAINING", "DatasetSplit"),
            self.cfg.GetStringValue("CGANTRAINING", "LRScheduler"),
            self.cfg.GetFloatValue("CGANTRAINING", "LearningRateDiscriminator"),
            self.cfg.GetFloatValue("CGANTRAINING", "LearningRateGenerator"),
            self.cfg.GetBoolValue("CGANTRAINING", "FormatImages"),
            self.cfg.GetBoolValue("CGANTRAINING", "TrackModeCollapse"),
            self.cfg.GetIntValue("CGANTRAINING", "ModeCollpseThreshold"),
            )

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

        self.classifier = cf.ClassifierMLModel(
            self.cfg.GetIntValue("CLASSIFIERTRAINING", "BatchSize"),
            self.cfg.GetIntValue("CGANTRAINING", "ImageChannels"),
            self.NumberOfClasses,
            self.cfg.GetIntValue("CLASSIFIERTRAINING", "ImageSize"),
            self.cfg.GetIntValue("CLASSIFIERTRAINING", "EpochCount"),
            self.cfg.GetIntValue("CLASSIFIERTRAINING", "RefreshUIEachXIteration"),
            self.cfg.GetStringValue("CLASSIFIERTRAINING", "TrainDatasetDir"),
            self.cfg.GetStringValue("CLASSIFIEROUTPUT", "ClassifyDir"),
            self.cfg.GetStringValue("CLASSIFIEROUTPUT", "LogDir"),
            self.cfg.GetBoolValue("CLASSIFIERTRAINING", "SaveCheckpoints"),
            self.cfg.GetBoolValue("CLASSIFIERTRAINING", "UseSavedModel"),
            self.cfg.GetStringValue("CLASSIFIERTRAINING", "CheckpointPath"),
            self.cfg.GetStringValue("CLASSIFIERTRAINING", "LatestCheckpointPath"),
            self.cfg.GetStringValue("CLASSIFIERTRAINING", "LogPath"),
            self.cfg.GetFloatValue("CLASSIFIERTRAINING", "DatasetSplit"),
            self.cfg.GetStringValue("CLASSIFIERTRAINING", "LRScheduler"),
            self.cfg.GetFloatValue("CLASSIFIERTRAINING", "LearningRateClassifier"),
            self.cfg.GetBoolValue("CLASSIFIERTRAINING", "FormatImages"),
            self.cfg.GetBoolValue("CLASSIFIEROUTPUT", "FormatImages")
            )

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
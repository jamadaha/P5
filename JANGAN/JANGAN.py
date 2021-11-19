import importlib
import os

from importlib import reload

from ProjectTools import ConfigHelper
 
import CGAN as cg
import DataGenerator as dg
import Classifier as cf
import Classifier.DataLoader

class JANGAN():
    cfg = None
    cgan = None
    classifier = None

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
        hf.DeleteFolderAndAllContents(self.cfg.GetStringValue("DATAGENERATOR","OutputLettersPath"))

        print(f" --- Done! --- ")

    def MakeCGANDataset(self):
        if self.cfg.GetBoolValue("DATAGENERATOR", "PurgePreviousData"):
            self.PurgeRunDataFolder()

        print(" --- Generating dataset if not there --- ")

        datagen = dg.DataGenerator()
        datagen.ConfigureFileImporter(
            self.cfg.GetStringValue("DATAGENERATOR","TextPath"),
            self.cfg.GetStringValue("DATAGENERATOR","LetterDownloadURL"),
            self.cfg.GetJsonValue("DATAGENERATOR","TextDownloadURLS"),
            self.cfg.GetStringValue("DATAGENERATOR","TempDownloadLetterPath"),
            self.cfg.GetStringValue("DATAGENERATOR", "TempDownloadLetterFileName"))
        datagen.ConfigureTextSequence(
            self.cfg.GetStringValue("DATAGENERATOR", "TextPath"))
        datagen.ConfigureDataExtractor(
            self.cfg.GetStringValue("DATAGENERATOR", "OutputLettersPath"),
            self.cfg.GetStringValue("DATAGENERATOR", "TempDownloadLetterPath"),
            self.cfg.GetStringValue("DATAGENERATOR", "TempDownloadLetterFileName"),
            self.cfg.GetIntValue("DATAGENERATOR", "MinimumLetterCount"),
            self.cfg.GetIntValue("DATAGENERATOR", "MaximumLetterCount"),
            self.cfg.GetStringValue("DATAGENERATOR", "OutputLetterFormat"),
            self.cfg.GetBoolValue("DATAGENERATOR", "IncludeNumbers"),
            self.cfg.GetBoolValue("DATAGENERATOR", "IncludeLetters"))
        datagen.GenerateData()

        print(" --- Done! --- ")
        print("")

    def TrainCGAN(self):
        print(" --- Training CGAN --- ")

        classCount = 0
        for entry in os.scandir(self.cfg.GetStringValue("DATAGENERATOR", "OutputLettersPath")):
            if entry.is_dir():
                classCount += 1

        self.cgan = cg.CGAN(
            self.cfg.GetIntValue("CGAN", "BatchSize"),
            1,
            classCount,
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
            self.cfg.GetStringValue("CGAN", "LogPath"),
            self.cfg.GetFloatValue("CGAN", "DatasetSplit"),
            self.cfg.GetFloatValue("CGAN", "AccuracyThreshold"))

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

        self.classifier = cf.Classifier(
            self.cfg.GetIntValue("Classifier", "Epochs"),
            self.cfg.GetBoolValue("Classifier", "Retrain"),
            self.cfg.GetStringValue("Classifier", "ModelName"),
            self.cfg.GetStringValue("Classifier", "ModelPath"),
            self.cfg.GetStringValue("CGAN", "OutputDir"),
            self.cfg.GetIntValue("Classifier", "BatchSize"),
            self.cfg.GetIntValue("Classifier", "ImageHeight"),
            self.cfg.GetIntValue("Classifier", "ImageWidth"),
            self.cfg.GetIntValue("Classifier", "Seed"),
            self.cfg.GetIntValue("Classifier", "Split")
            )

        #Mount data from GAN
        self.classifier.LoadData()

        #Train model
        self.classifier.TrainClassifier(data)
        
        # Produce output
        vdata = dataLoader.LoadDataSet(self.cfg.GetStringValue("Classifier", "ValidationData"), self.cfg.GetStringValue("ValidationSplit"), self.cfg.GetStringValue("Classifier", "Subset"), self.cfg.GetIntValue("Classifier", "Seed"))

        self.classifier.ProduceStatistics(
            vdata,
            self.cfg.GetFloatValue("Classifier", "ValidationSplit"),
            self.cfg.GetStringValue("Classifier", "Subset"),
            self.cfg.GetIntValue("Classifier", "Seed"))

        print(" --- Done! --- ")
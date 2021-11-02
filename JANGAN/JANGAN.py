import importlib
from importlib import reload

from ProjectTools import ConfigHelper
 
import CGAN as cg
import DataGenerator as dg

class JANGAN():
    cfg = None

    def __init__(self, expFile, configFile):
        importlib.import_module(expFile)
        self.LoadConfig(configFile)

    def LoadConfig(self, configFile):
        print(" --- Loading experiment config file --- ")
        self.cfg = ConfigHelper.ConfigHelper(configFile)
        self.cfg.LoadConfig()
        print(" --- Done! --- ")
        print("")

    def Run(self):
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
            self.cfg.GetStringValue("DATAGENERATOR", "OutputLetterFormat"))
        datagen.GenerateData()

        print(" --- Done! --- ")
        print("")
        print(" --- Training CGAN --- ")

        cgan = cg.CGAN(
            self.cfg.GetIntValue("CGAN", "BatchSize"),
            1,
            self.cfg.GetIntValue("CGAN", "NumberOfClasses"),
            self.cfg.GetIntValue("CGAN", "ImageSize"),
            self.cfg.GetIntValue("CGAN", "LatentDimension"),
            self.cfg.GetIntValue("CGAN", "EpochCount"),
            self.cfg.GetIntValue("CGAN", "RefreshUIEachXIteration"),
            self.cfg.GetIntValue("CGAN", "NumberOfFakeImagesToOutput"),
            self.cfg.GetStringValue("CGAN", "TrainDatasetDir"),
            self.cfg.GetStringValue("CGAN", "TestDatasetDir"))

        cgan.SetupCGAN()
        cgan.LoadDataset()
        cgan.TrainGAN()
        cgan.ProduceLetters()

        print(" --- Done! --- ")
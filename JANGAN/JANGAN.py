import importlib
from importlib import reload

from ProjectTools import ConfigHelper
 
import CGAN as cg
import DataGenerator as dg

class JANGAN():
    cfg = None
    cgan = None

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
        import os, shutil

        folder = self.cfg.GetStringValue("DATAGENERATOR","OutputLettersPath")

        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            shutil.rmtree(folder)

        print(f" --- Done! --- ")

    def Run(self):
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
        print(" --- Training CGAN --- ")

        self.cgan = cg.CGAN(
            self.cfg.GetIntValue("CGAN", "BatchSize"),
            1,
            self.cfg.GetIntValue("CGAN", "NumberOfClasses"),
            self.cfg.GetIntValue("CGAN", "ImageSize"),
            self.cfg.GetIntValue("CGAN", "LatentDimension"),
            self.cfg.GetIntValue("CGAN", "EpochCount"),
            self.cfg.GetIntValue("CGAN", "RefreshUIEachXIteration"),
            self.cfg.GetIntValue("CGAN", "NumberOfFakeImagesToOutput"),
            self.cfg.GetStringValue("CGAN", "TrainDatasetDir"),
            self.cfg.GetStringValue("CGAN", "TestDatasetDir"),
            self.cfg.GetBoolValue("CGAN", "SaveCheckpoints"),
            self.cfg.GetBoolValue("CGAN", "UseSavedModel"))

        self.cgan.SetupCGAN()
        self.cgan.LoadDataset()
        self.cgan.TrainGAN()

        print(" --- Done! --- ")

    def ProduceOutput(self):
        print(" --- Producing output --- ")

        self.cgan.ProduceLetters()

        print(" --- Done! --- ")
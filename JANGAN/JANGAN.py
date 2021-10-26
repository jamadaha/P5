from ProjectTools import ConfigHelper
import os

print(" --- Loading config files --- ")
cfg = ConfigHelper.ConfigHelper()
cfg.LoadConfig()
print(" --- Done! --- ")
print("")

print(" --- Checking, installing and updating packages --- ")

if cfg.GetStringValue("GENERAL","AutoUpdatePackages") == "True":
    os.environ["AutoPackageInstaller_AutoUpdate"] = "True"
else:
    os.environ["AutoPackageInstaller_AutoUpdate"] = "False"

import CGAN as cg
import DataGenerator as dg

print(" --- Done! --- ")
print("")

print(" --- Generating dataset if not there --- ")

datagen = dg.DataGenerator()
datagen.ConfigureFileImporter(
    cfg.GetStringValue("DATAGENERATOR","TextPath"),
    cfg.GetStringValue("DATAGENERATOR","LetterDownloadURL"),
    cfg.GetJsonValue("DATAGENERATOR","TextDownloadURLS"),
    cfg.GetStringValue("DATAGENERATOR","TempDownloadLetterPath"),
    cfg.GetStringValue("DATAGENERATOR", "TempDownloadLetterFileName"))
datagen.ConfigureTextSequence(
    cfg.GetStringValue("DATAGENERATOR", "TextPath"))
datagen.ConfigureDataExtractor(
    cfg.GetStringValue("DATAGENERATOR", "OutputLettersPath"),
    cfg.GetStringValue("DATAGENERATOR", "TempDownloadLetterPath"),
    cfg.GetStringValue("DATAGENERATOR", "TempDownloadLetterFileName"),
    cfg.GetIntValue("DATAGENERATOR", "MinimumLetterCount"),
    cfg.GetIntValue("DATAGENERATOR", "MaximumLetterCount"),
    cfg.GetStringValue("DATAGENERATOR", "OutputLetterFormat"))
datagen.GenerateData()

print(" --- Done! --- ")
print("")
print(" --- Training CGAN --- ")

cgan = cg.CGAN(
    cfg.GetIntValue("CGAN", "BatchSize"),
    1,
    cfg.GetIntValue("CGAN", "NumberOfClasses"),
    cfg.GetIntValue("CGAN", "ImageSize"),
    cfg.GetIntValue("CGAN", "LatentDimension"),
    cfg.GetIntValue("CGAN", "EpochCount"),
    cfg.GetIntValue("CGAN", "RefreshUIEachXIteration"),
    cfg.GetIntValue("CGAN", "NumberOfFakeImagesToOutput"),
    cfg.GetStringValue("CGAN", "TrainDatasetDir"),
    cfg.GetStringValue("CGAN", "TestDatasetDir"))

cgan.SetupCGAN()
cgan.LoadDataset()
cgan.TrainGAN()
cgan.ProduceLetters()

print(" --- Done! --- ")
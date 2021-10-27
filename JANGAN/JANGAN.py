from ProjectTools import ConfigHelper
import os

print(" --- Loading config files --- ")
cfg = ConfigHelper.ConfigHelper()
cfg.LoadConfig()
print(" --- Done! --- ")
print("")

print(" --- Setting up Auto Package Installer --- ")

from ProjectTools import AutoPackageInstaller as ap

if cfg.GetStringValue("AUTOPACKAGEINSTALLER","AutoUpdatePackages") == "True":
    ap.AutoUpdate = True

ap.InstallAllMissingModules = cfg.GetStringValue("AUTOPACKAGEINSTALLER","YesToAllInstalls")
ap.UpdateAllModules = cfg.GetStringValue("AUTOPACKAGEINSTALLER","YesToAllUpdates")

# This is for GitHub Actions
if os.getenv("AutoPackageInstaller_YesToAllModules"):
    if os.environ["AutoPackageInstaller_YesToAllModules"] == "True":
        ap.InstallAllMissingModules = True

print(" --- Done! --- ")
print("")

import CGAN as cg
import DataGenerator as dg

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
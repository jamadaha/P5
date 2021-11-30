from ProjectTools import ConfigHelper as cfgh

class JANGANConfigChecker():
    ThrowIfConfigFileBad = True
    __CheckedKeyCount = 0

    def CheckConfig(self, cfg : cfgh.ConfigHelper, throwIfConfigFileBad):
        self.ThrowIfConfigFileBad = throwIfConfigFileBad
        print(" --- Checking config file integrity --- ")

        # Data Generator
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "DATAGENERATOR", "BasePath")
        self.__CheckKey(cfg, "DATAGENERATOR", "TextPath")
        self.__CheckKey(cfg, "DATAGENERATOR", "LetterPath")
        self.__CheckKey(cfg, "DATAGENERATOR", "LetterDownloadPath")
        self.__CheckKey(cfg, "DATAGENERATOR", "LetterDownloadName")
        self.__CheckKey(cfg, "DATAGENERATOR", "LetterDownloadURL")
        self.__CheckKey(cfg, "DATAGENERATOR", "TextDownloadURLS")
        self.__CheckKey(cfg, "DATAGENERATOR", "PurgePreviousData")
        self.__CheckKey(cfg, "DATAGENERATOR", "IncludeNumbers")
        self.__CheckKey(cfg, "DATAGENERATOR", "IncludeLetters")
        self.__CheckKey(cfg, "DATAGENERATOR", "DistributionPath")
        self.__CheckKey(cfg, "DATAGENERATOR", "PrintDistribution")
        self.__CheckKey(cfg, "DATAGENERATOR", "LetterOutputFormat")
        self.__CheckKey(cfg, "DATAGENERATOR", "MinimumLetterCount")
        self.__CheckKey(cfg, "DATAGENERATOR", "MaximumLetterCount")
        self.__CheckKeyCount(cfg, "DATAGENERATOR")

        # CGAN
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "CGAN", "TrainDatasetDir")
        self.__CheckKey(cfg, "CGAN", "TestDatasetDir")
        self.__CheckKey(cfg, "CGAN", "OutputDir")
        self.__CheckKey(cfg, "CGAN", "ConfigCopyPath")
        self.__CheckKey(cfg, "CGAN", "ImageSize")
        self.__CheckKey(cfg, "CGAN", "BatchSize")
        self.__CheckKey(cfg, "CGAN", "LatentDimension")
        self.__CheckKey(cfg, "CGAN", "EpochCount")
        self.__CheckKey(cfg, "CGAN", "RefreshUIEachXIteration")
        self.__CheckKey(cfg, "CGAN", "NumberOfFakeImagesToOutput")
        self.__CheckKey(cfg, "CGAN", "CheckpointPath")
        self.__CheckKey(cfg, "CGAN", "LatestCheckpointPath")
        self.__CheckKey(cfg, "CGAN", "SaveCheckpoints")
        self.__CheckKey(cfg, "CGAN", "UseSavedModel")
        self.__CheckKey(cfg, "CGAN", "LogPath")
        self.__CheckKey(cfg, "CGAN", "DatasetSplit")
        self.__CheckKey(cfg, "CGAN", "LRScheduler")
        self.__CheckKey(cfg, "CGAN", "LearningRateDiscriminator")
        self.__CheckKey(cfg, "CGAN", "LearningRateGenerator")
        self.__CheckKeyCount(cfg, "CGAN")

        # Classifier
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "Classifier", "TrainDatasetDir")
        self.__CheckKey(cfg, "Classifier", "TestDatasetDir")
        self.__CheckKey(cfg, "Classifier", "ClassifyDir")
        self.__CheckKey(cfg, "Classifier", "ConfigCopyPath")
        self.__CheckKey(cfg, "Classifier", "ImageSize")
        self.__CheckKey(cfg, "Classifier", "BatchSize")
        self.__CheckKey(cfg, "Classifier", "EpochCount")
        self.__CheckKey(cfg, "Classifier", "RefreshUIEachXIteration")
        self.__CheckKey(cfg, "Classifier", "CheckpointPath")
        self.__CheckKey(cfg, "Classifier", "LatestCheckpointPath")
        self.__CheckKey(cfg, "Classifier", "SaveCheckpoints")
        self.__CheckKey(cfg, "Classifier", "UseSavedModel")
        self.__CheckKey(cfg, "Classifier", "LogPath")
        self.__CheckKey(cfg, "Classifier", "DatasetSplit")
        self.__CheckKey(cfg, "Classifier", "AccuracyThreshold")
        self.__CheckKey(cfg, "Classifier", "LRScheduler")
        self.__CheckKey(cfg, "Classifier", "LearningRateClassifier")
        self.__CheckKeyCount(cfg, "Classifier")

        print(" --- Done! --- ")

    def __CheckKey(self, cfg : cfgh.ConfigHelper, category, key):
        if self.ThrowIfConfigFileBad == True:
            cfg.CheckIfKeyExists(category, key)
        else:
            try:
                cfg.CheckIfKeyExists(category, key)
            except (cfgh.CategoryNotFoundException, cfgh.KeyNotFoundException) as e:
                print(f"Warning! A config key/category is missing! (category: {category}, key: {key})")
        self.__CheckedKeyCount += 1

    def __CheckKeyCount(self, cfg : cfgh.ConfigHelper, category):
        keyCount = cfg.CategoryKeyCount(category)
        if keyCount != self.__CheckedKeyCount:
            if self.ThrowIfConfigFileBad == True:
                raise Exception(f"Error! Config category '{category}' key count did not match the expected! (expected: {keyCount}, actual {self.__CheckedKeyCount})")
            else:
                print(f"Warning! Config category '{category}' key count did not match the expected! (expected: {keyCount}, actual {self.__CheckedKeyCount})")
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

        # CGAN Training
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "CGANTraining", "TrainDatasetDir")
        self.__CheckKey(cfg, "CGANTraining", "TestDatasetDir")
        self.__CheckKey(cfg, "CGANTraining", "ConfigCopyPath")
        self.__CheckKey(cfg, "CGANTraining", "ImageSize")
        self.__CheckKey(cfg, "CGANTraining", "BatchSize")
        self.__CheckKey(cfg, "CGANTraining", "LatentDimension")
        self.__CheckKey(cfg, "CGANTraining", "EpochCount")
        self.__CheckKey(cfg, "CGANTraining", "RefreshUIEachXIteration")
        self.__CheckKey(cfg, "CGANTraining", "CheckpointPath")
        self.__CheckKey(cfg, "CGANTraining", "LatestCheckpointPath")
        self.__CheckKey(cfg, "CGANTraining", "SaveCheckpoints")
        self.__CheckKey(cfg, "CGANTraining", "UseSavedModel")
        self.__CheckKey(cfg, "CGANTraining", "LogPath")
        self.__CheckKey(cfg, "CGANTraining", "DatasetSplit")
        self.__CheckKey(cfg, "CGANTraining", "LRScheduler")
        self.__CheckKey(cfg, "CGANTraining", "LearningRateDiscriminator")
        self.__CheckKey(cfg, "CGANTraining", "LearningRateGenerator")
        self.__CheckKeyCount(cfg, "CGANTraining")

        # CGAN Output
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "CGANOutput", "OutputDir")
        self.__CheckKey(cfg, "CGANOutput", "NumberOfFakeImagesToOutput")
        self.__CheckKeyCount(cfg, "CGANOutput")

        # Classifier Training
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "ClassifierTraining", "TrainDatasetDir")
        self.__CheckKey(cfg, "ClassifierTraining", "TestDatasetDir")
        self.__CheckKey(cfg, "ClassifierTraining", "ConfigCopyPath")
        self.__CheckKey(cfg, "ClassifierTraining", "ImageSize")
        self.__CheckKey(cfg, "ClassifierTraining", "BatchSize")
        self.__CheckKey(cfg, "ClassifierTraining", "EpochCount")
        self.__CheckKey(cfg, "ClassifierTraining", "RefreshUIEachXIteration")
        self.__CheckKey(cfg, "ClassifierTraining", "CheckpointPath")
        self.__CheckKey(cfg, "ClassifierTraining", "LatestCheckpointPath")
        self.__CheckKey(cfg, "ClassifierTraining", "SaveCheckpoints")
        self.__CheckKey(cfg, "ClassifierTraining", "UseSavedModel")
        self.__CheckKey(cfg, "ClassifierTraining", "LogPath")
        self.__CheckKey(cfg, "ClassifierTraining", "DatasetSplit")
        self.__CheckKey(cfg, "ClassifierTraining", "AccuracyThreshold")
        self.__CheckKey(cfg, "ClassifierTraining", "LRScheduler")
        self.__CheckKey(cfg, "ClassifierTraining", "LearningRateClassifier")
        self.__CheckKeyCount(cfg, "ClassifierTraining")

        # Classifier Output
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "ClassifierOutput", "ClassifyDir")
        self.__CheckKeyCount(cfg, "ClassifierOutput")

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
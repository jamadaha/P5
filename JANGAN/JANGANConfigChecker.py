from ProjectTools import ConfigHelper as cfgh

class JANGANConfigChecker():
    ThrowIfConfigFileBad = True
    __CheckedKeyCount = 0

    def CheckConfig(self, cfg : cfgh.ConfigHelper, throwIfConfigFileBad):
        self.ThrowIfConfigFileBad = throwIfConfigFileBad
        print(" --- Checking config file integrity --- ")

        # Global
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "GLOBAL", "ConfigCopyPath")
        self.__CheckKeyCount(cfg, "GLOBAL")

        # CGAN Data Generator
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "BasePath")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "TextPath")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "LetterPath")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "LetterDownloadPath")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "LetterDownloadName")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "LetterDownloadURL")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "TextDownloadURLS")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "PurgePreviousData")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "IncludeNumbers")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "IncludeLetters")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "DistributionPath")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "PrintDistribution")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "LetterOutputFormat")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "MinimumLetterCount")
        self.__CheckKey(cfg, "CGANDATAGENERATOR", "MaximumLetterCount")
        self.__CheckKeyCount(cfg, "CGANDATAGENERATOR")

        # Classifier Data Generator
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "BasePath")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "TextPath")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "LetterPath")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "LetterDownloadPath")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "LetterDownloadName")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "LetterDownloadURL")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "TextDownloadURLS")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "PurgePreviousData")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "IncludeNumbers")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "IncludeLetters")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "DistributionPath")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "PrintDistribution")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "LetterOutputFormat")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "MinimumLetterCount")
        self.__CheckKey(cfg, "ClassifierDATAGENERATOR", "MaximumLetterCount")
        self.__CheckKeyCount(cfg, "ClassifierDATAGENERATOR")

        # CGAN Training
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "CGANTraining", "TrainDatasetDir")
        self.__CheckKey(cfg, "CGANTraining", "TestDatasetDir")
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
        self.__CheckKey(cfg, "ClassifierOutput", "LogDir")
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
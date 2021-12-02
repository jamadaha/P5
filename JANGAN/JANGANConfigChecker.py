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
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "BasePath")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "TextPath")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "LetterPath")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "LetterDownloadPath")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "LetterDownloadName")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "LetterDownloadURL")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "TextDownloadURLS")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "PurgePreviousData")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "IncludeNumbers")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "IncludeLetters")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "DistributionPath")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "PrintDistribution")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "LetterOutputFormat")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "MinimumLetterCount")
        self.__CheckKey(cfg, "CLASSIFIERDATAGENERATOR", "MaximumLetterCount")
        self.__CheckKeyCount(cfg, "CLASSIFIERDATAGENERATOR")

        # CGAN Training
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "CGANTRAINING", "TrainDatasetDir")
        self.__CheckKey(cfg, "CGANTRAINING", "TestDatasetDir")
        self.__CheckKey(cfg, "CGANTRAINING", "ImageSize")
        self.__CheckKey(cfg, "CGANTRAINING", "BatchSize")
        self.__CheckKey(cfg, "CGANTRAINING", "LatentDimension")
        self.__CheckKey(cfg, "CGANTRAINING", "EpochCount")
        self.__CheckKey(cfg, "CGANTRAINING", "RefreshUIEachXIteration")
        self.__CheckKey(cfg, "CGANTRAINING", "CheckpointPath")
        self.__CheckKey(cfg, "CGANTRAINING", "LatestCheckpointPath")
        self.__CheckKey(cfg, "CGANTRAINING", "SaveCheckpoints")
        self.__CheckKey(cfg, "CGANTRAINING", "UseSavedModel")
        self.__CheckKey(cfg, "CGANTRAINING", "LogPath")
        self.__CheckKey(cfg, "CGANTRAINING", "EpochImgDir")
        self.__CheckKey(cfg, "CGANTRAINING", "DatasetSplit")
        self.__CheckKey(cfg, "CGANTRAINING", "LRScheduler")
        self.__CheckKey(cfg, "CGANTRAINING", "LearningRateDiscriminator")
        self.__CheckKey(cfg, "CGANTRAINING", "LearningRateGenerator")
        self.__CheckKeyCount(cfg, "CGANTRAINING")

        # CGAN Output
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "CGANOUTPUT", "OutputDir")
        self.__CheckKey(cfg, "CGANOUTPUT", "NumberOfFakeImagesToOutput")
        self.__CheckKeyCount(cfg, "CGANOUTPUT")

        # Classifier Training
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "TrainDatasetDir")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "TestDatasetDir")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "ImageSize")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "BatchSize")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "EpochCount")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "RefreshUIEachXIteration")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "CheckpointPath")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "LatestCheckpointPath")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "SaveCheckpoints")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "UseSavedModel")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "LogPath")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "DatasetSplit")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "AccuracyThreshold")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "LRScheduler")
        self.__CheckKey(cfg, "CLASSIFIERTRAINING", "LearningRateClassifier")
        self.__CheckKeyCount(cfg, "CLASSIFIERTRAINING")

        # Classifier Output
        self.__CheckedKeyCount = 0
        self.__CheckKey(cfg, "CLASSIFIEROUTPUT", "ClassifyDir")
        self.__CheckKey(cfg, "CLASSIFIEROUTPUT", "LogDir")
        self.__CheckKey(cfg, "CLASSIFIEROUTPUT", "FormatImages")
        self.__CheckKeyCount(cfg, "CLASSIFIEROUTPUT")

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
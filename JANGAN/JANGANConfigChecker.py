from ProjectTools import BaseConfigChecker as bcc

class JANGANConfigChecker(bcc.BaseConfigChecker):
    def CheckConfig(self):
        print(" --- Checking config file integrity --- ")

        # Global
        self.CheckedKeyCount = 0
        self.CheckKey("GLOBAL", "ConfigCopyPath")
        self.CheckKey("GLOBAL", "BasePath")
        self.CheckKeyCount("GLOBAL")

        # Data generator General
        self.CheckedKeyCount = 0
        self.CheckKey("DATAGENERATOR", "TextPath")
        self.CheckKey("DATAGENERATOR", "LetterDownloadPath")
        self.CheckKey("DATAGENERATOR", "LetterDownloadName")
        self.CheckKey("DATAGENERATOR", "LetterDownloadURL")
        self.CheckKey("DATAGENERATOR", "TextDownloadURLS")
        self.CheckKeyCount("DATAGENERATOR")

        # CGAN Data Generator
        self.CheckedKeyCount = 0
        self.CheckKey("CGANDATAGENERATOR", "OutputPath")
        self.CheckKey("CGANDATAGENERATOR", "LetterOutputFormat")
        self.CheckKey("CGANDATAGENERATOR", "PurgePreviousData")
        self.CheckKey("CGANDATAGENERATOR", "IncludeNumbers")
        self.CheckKey("CGANDATAGENERATOR", "IncludeLetters")
        self.CheckKey("CGANDATAGENERATOR", "DistributionPath")
        self.CheckKey("CGANDATAGENERATOR", "PrintDistribution")
        self.CheckKey("CGANDATAGENERATOR", "MinimumLetterCount")
        self.CheckKey("CGANDATAGENERATOR", "MaximumLetterCount")
        self.CheckKeyCount("CGANDATAGENERATOR")

        # Classifier Data Generator
        self.CheckedKeyCount = 0
        self.CheckKey("CLASSIFIERDATAGENERATOR", "OutputPath")
        self.CheckKey("CLASSIFIERDATAGENERATOR", "LetterOutputFormat")
        self.CheckKey("CLASSIFIERDATAGENERATOR", "PurgePreviousData")
        self.CheckKey("CLASSIFIERDATAGENERATOR", "IncludeNumbers")
        self.CheckKey("CLASSIFIERDATAGENERATOR", "IncludeLetters")
        self.CheckKey("CLASSIFIERDATAGENERATOR", "DistributionPath")
        self.CheckKey("CLASSIFIERDATAGENERATOR", "PrintDistribution")
        self.CheckKey("CLASSIFIERDATAGENERATOR", "MinimumLetterCount")
        self.CheckKey("CLASSIFIERDATAGENERATOR", "MaximumLetterCount")
        self.CheckKeyCount("CLASSIFIERDATAGENERATOR")

        # CGAN Training
        self.CheckedKeyCount = 0
        self.CheckKey("CGANTRAINING", "TrainDatasetDir")
        self.CheckKey("CGANTRAINING", "ImageSize")
        self.CheckKey("CGANTRAINING", "ImageChannels")
        self.CheckKey("CGANTRAINING", "BatchSize")
        self.CheckKey("CGANTRAINING", "LatentDimension")
        self.CheckKey("CGANTRAINING", "EpochCount")
        self.CheckKey("CGANTRAINING", "RefreshUIEachXIteration")
        self.CheckKey("CGANTRAINING", "CheckpointPath")
        self.CheckKey("CGANTRAINING", "LatestCheckpointPath")
        self.CheckKey("CGANTRAINING", "SaveCheckpoints")
        self.CheckKey("CGANTRAINING", "UseSavedModel")
        self.CheckKey("CGANTRAINING", "LogPath")
        self.CheckKey("CGANTRAINING", "EpochImgDir")
        self.CheckKey("CGANTRAINING", "DatasetSplit")
        self.CheckKey("CGANTRAINING", "LRScheduler")
        self.CheckKey("CGANTRAINING", "LearningRateDiscriminator")
        self.CheckKey("CGANTRAINING", "LearningRateGenerator")
        self.CheckKey("CGANTRAINING", "FormatImages")
        self.CheckKey("CGANTRAINING", "TrackModeCollapse")
        self.CheckKey("CGANTRAINING", "ModeCollpseThreshold")
        self.CheckKeyCount("CGANTRAINING")

        # CGAN Output
        self.CheckedKeyCount = 0
        self.CheckKey("CGANOUTPUT", "OutputDir")
        self.CheckKey("CGANOUTPUT", "NumberOfFakeImagesToOutput")
        self.CheckKeyCount("CGANOUTPUT")

        # Classifier Training
        self.CheckedKeyCount = 0
        self.CheckKey("CLASSIFIERTRAINING", "TrainDatasetDir")
        self.CheckKey("CLASSIFIERTRAINING", "ImageSize")
        self.CheckKey("CLASSIFIERTRAINING", "ImageChannels")
        self.CheckKey("CLASSIFIERTRAINING", "BatchSize")
        self.CheckKey("CLASSIFIERTRAINING", "EpochCount")
        self.CheckKey("CLASSIFIERTRAINING", "RefreshUIEachXIteration")
        self.CheckKey("CLASSIFIERTRAINING", "CheckpointPath")
        self.CheckKey("CLASSIFIERTRAINING", "LatestCheckpointPath")
        self.CheckKey("CLASSIFIERTRAINING", "SaveCheckpoints")
        self.CheckKey("CLASSIFIERTRAINING", "UseSavedModel")
        self.CheckKey("CLASSIFIERTRAINING", "LogPath")
        self.CheckKey("CLASSIFIERTRAINING", "DatasetSplit")
        self.CheckKey("CLASSIFIERTRAINING", "LRScheduler")
        self.CheckKey("CLASSIFIERTRAINING", "LearningRateClassifier")
        self.CheckKey("CLASSIFIERTRAINING", "FormatImages")
        self.CheckKeyCount("CLASSIFIERTRAINING")

        # Classifier Output
        self.CheckedKeyCount = 0
        self.CheckKey("CLASSIFIEROUTPUT", "ClassifyDir")
        self.CheckKey("CLASSIFIEROUTPUT", "LogDir")
        self.CheckKey("CLASSIFIEROUTPUT", "FormatImages")
        self.CheckKeyCount("CLASSIFIEROUTPUT")

        print(" --- Done! --- ")

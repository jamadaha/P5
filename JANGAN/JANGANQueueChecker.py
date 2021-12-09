from ProjectTools import BaseConfigChecker as bcc

class JANGANQueueChecker(bcc.BaseConfigChecker):
    def CheckConfig(self):
        self.CheckedKeyCount = 0
        self.CheckKey("EXPERIMENTS", "ThrowIfConfigFileBad")
        self.CheckKey("EXPERIMENTS", "ExperimentList")
        self.CheckKeyCount("EXPERIMENTS")

        expDict = self.cfg.GetListValue("EXPERIMENTS", "ExperimentList")
        for key in expDict:
            self.CheckedKeyCount = 0
            self.CheckKey(key, "ModuleName")
            self.CheckKey(key, "AmountOfTimesToRun")
            self.CheckKey(key, "ConfigFile")
            self.CheckKey(key, "MakeCGANDataset")
            self.CheckKey(key, "MakeClassifierDataset")
            self.CheckKey(key, "TrainCGAN")
            self.CheckKey(key, "ProduceCGANLetters")
            self.CheckKey(key, "TrainClassifier")
            self.CheckKey(key, "ClassifyImages")
            self.CheckKeyCount(key)

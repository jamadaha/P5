from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import ConfigHelper    
import JANGANQueueChecker
import JANGANModuleReloader

ap.CheckAndInstall("traceback")

import traceback

print(" --- Loading queue config file --- ")
cfg = ConfigHelper.ConfigHelper("ExperimentQueueConfig.ini")
cfg.LoadConfig()
throwIfConfigFileBad = cfg.GetBoolValue("EXPERIMENTS", "ThrowIfConfigFileBad")
print(" --- Done! --- ")
print("")

print(" --- Checking the queue file --- ")
queueChecker = JANGANQueueChecker.JANGANQueueChecker(cfg, throwIfConfigFileBad)
queueChecker.CheckConfig()
print(" --- Done! --- ")
print("")

expDict = cfg.GetListValue("EXPERIMENTS","ExperimentList")
for key in expDict:
    count = cfg.GetIntValue(key,'AmountOfTimesToRun')
    for n in range(count):
        print("")
        print(f" --- Running experiment '{key}' --- ")
        print(f" --- Iteration {n + 1} out of {count} --- ")
        print("")

        try:
            import JANGAN as jg
            expJANGAN = jg.JANGAN(key, cfg.GetStringValue(key, 'ModuleName'), cfg.GetStringValue(key, 'ConfigFile'), throwIfConfigFileBad)
            if cfg.GetBoolValue(key, 'MakeCGANDataset') == True:
                expJANGAN.MakeCGANDataset()
            if cfg.GetBoolValue(key, 'MakeClassifierDataset') == True:
                expJANGAN.MakeClassifyerDataset()
            if cfg.GetBoolValue(key, 'TrainCGAN') == True:
                expJANGAN.TrainCGAN()
            if cfg.GetBoolValue(key, 'ProduceCGANLetters') == True:
                expJANGAN.ProduceOutput()
            if cfg.GetBoolValue(key, 'TrainClassifier') == True:
                expJANGAN.TrainClassifier()
            if cfg.GetBoolValue(key, 'ClassifyImages') == True:
                expJANGAN.ClassifyCGANOutput()


        except Exception as e:
            print("")
            print(f"      ERROR! Experiment '{key}' failed with error '{e}'")
            print( "      STACKTRACE")
            print(traceback.format_exc())
            print("")

        JANGANModuleReloader.JANGANModuleReloader().ReloadModules()

        print("")
        print(f" --- Experiment iteration '{n + 1}' done! --- ")
        print("")

    print("")
    print(f" --- Experiment '{key}' done! --- ")
    print("")
import traceback
from ProjectTools import ConfigHelper    
import JANGANQueueChecker
import JANGANModuleReloader

print(" --- Loading queue config file --- ")
cfg = ConfigHelper.ConfigHelper("ExperimentQueueConfig.ini")
cfg.LoadConfig()
print(" --- Done! --- ")
print("")

print(" --- Checking the queue file --- ")
queueChecker = JANGANQueueChecker.JANGANQueueChecker(cfg)
queueChecker.CheckConfig()
print(" --- Done! --- ")

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
            expJANGAN = jg.JANGAN(cfg.GetStringValue(key, 'ModuleName'), cfg.GetStringValue(key, 'ConfigFile'))
            if cfg.GetBoolValue(key, 'MakeCGANDataset') == True:
                expJANGAN.MakeCGANDataset()
            if cfg.GetBoolValue(key, 'TrainCGAN') == True:
                expJANGAN.TrainCGAN()
            if cfg.GetBoolValue(key, 'ProduceCGANLetters') == True:
                expJANGAN.ProduceOutput()
            if expDict[key]['TrainClassifier'] == "True":
                expJANGAN.ClassifyCGANOutput()


        except Exception as e:
            print("")
            print(f"      ERROR! Experiment '{key}' failed with error '{e}'")
            print("      STACKTRACE")
            print(traceback.format_exc())
            print("")

        JANGANModuleReloader.JANGANModuleReloader().ReloadModules()

        print("")
        print(f" --- Experiment iteration '{n + 1}' done! --- ")
        print("")

    print("")
    print(f" --- Experiment '{key}' done! --- ")
    print("")
        

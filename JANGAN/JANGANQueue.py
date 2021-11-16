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

expDict = cfg.GetJsonValue("EXPERIMENTS","ExperimentList")
for key in expDict:
    count = expDict[key]['AmountOfTimesToRun']
    for n in range(count):
        print("")
        print(f" --- Running experiment '{key}' --- ")
        print(f" --- Iteration {n + 1} out of {count} --- ")
        print("")

        try:
            import JANGAN as jg

            expJANGAN = jg.JANGAN(expDict[key]['ModuleName'], expDict[key]['ConfigFile'])
            expJANGAN.ClassifyCGANOutput()
            if expDict[key]['MakeCGANDataset'] == "True":
                expJANGAN.MakeCGANDataset()
            if expDict[key]['TrainCGAn'] == "True":
                expJANGAN.TrainCGAN()
            if expDict[key]['ProduceCGANLetters'] == "True":
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
        

from importlib import reload
import traceback

def ReloadAllModules():
    print(" --- Reloading modules --- ")

    import CGAN as cg
    reload(cg.CGANKerasModel)
    reload(cg.CGANTrainer)
    reload(cg.DatasetFormatter)
    reload(cg.DatasetLoader)
    reload(cg.LayerDefinition)
    reload(cg.LetterProducer)
    reload(cg)

    import DataGenerator as dg
    reload(dg.DataExtractor)
    reload(dg.FileImporter)
    reload(dg.SharedFunctions)
    reload(dg.TextSequence)
    reload(dg)

    import JANGAN as jg
    reload(jg)

    print(" --- Done! --- ")

from ProjectTools import ConfigHelper    

print(" --- Loading queue config file --- ")
cfg = ConfigHelper.ConfigHelper("ExperimentQueueConfig.ini")
cfg.LoadConfig()
print(" --- Done! --- ")
print("")

expDict = cfg.GetJsonValue("EXPERIMENTS","ExperimentList");
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
            expJANGAN.Run()
            expJANGAN.ProduceOutput()

        except Exception as e:
            print("")
            print(f"      ERROR! Experiment '{key}' failed with error '{e}'")
            print("      STACKTRACE")
            print(traceback.format_exc())
            print("")

        print("")
        print(f" --- Experiment iteration '{n + 1}' done! --- ")
        print("")

    print("")
    print(f" --- Experiment '{key}' done! --- ")
    print("")
        

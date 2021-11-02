from importlib import reload

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

def PurgeRunDataFolder(folder):
    print(" --- Purging training data folder --- ")

    import os, shutil
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    shutil.rmtree(folder)

    print(f" --- Done! --- ")


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
        print("      ")
        print(f" --- Running experiment '{key}' --- ")
        print(f" --- Iteration {n + 1} out of {count} --- ")
        print("      ")

        try:
            import JANGAN as jg

            expJANGAN = jg.JANGAN(expDict[key]['ModuleName'], f"ExperimentQueue/{expDict[key]['ConfigFile']}")
            expJANGAN.Run()
        except Exception as e:
            print("      ")
            print(f"      ERROR! Experiment '{key}' failed with error '{e}'")
            print("      ")

        print("      ")
        print(f" --- Experiment '{key}' done! --- ")
        print("      ")

        ReloadAllModules();
        PurgeRunDataFolder(cfg.GetStringValue("DATAGENERATOR","BasePath"));

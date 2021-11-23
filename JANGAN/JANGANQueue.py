import traceback
import tracemalloc
from ProjectTools import ConfigHelper    
import JANGANQueueChecker
import JANGANModuleReloader
import CheckMemoryLeak as cml

print(" --- Loading queue config file --- ")
cfg = ConfigHelper.ConfigHelper("ExperimentQueueConfig.ini")
cfg.LoadConfig()
print(" --- Done! --- ")
print("")

print(" --- Checking the queue file --- ")
queueChecker = JANGANQueueChecker.JANGANQueueChecker(cfg)
queueChecker.CheckConfig()
print(" --- Done! --- ")
print("")

expDict = cfg.GetListValue("EXPERIMENTS","ExperimentList")

gCfg = ConfigHelper.ConfigHelper("GlobalConfig.ini")
gCfg.LoadConfig()

runTraceMalloc = gCfg.GetBoolValue("TRACEMALLOC", "RunTraceMalloc")

if (runTraceMalloc):
    print(" --- Starting Trace Malloc ---")
    tracemalloc.start()

    checkMemoryLeak = cml.CheckMemoryLeak()
    checkMemoryLeak.ConfigurePath(gCfg.GetStringValue("TRACEMALLOC", "TraceMallocDir"), gCfg.GetStringValue("TRACEMALLOC", "TraceMallocFile"))

for key in expDict:
    if (runTraceMalloc):
        checkMemoryLeak.SaveKey(key)

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

        except Exception as e:
            print("")
            print(f"      ERROR! Experiment '{key}' failed with error '{e}'")
            print( "      STACKTRACE")
            print(traceback.format_exc())
            print("")

        JANGANModuleReloader.JANGANModuleReloader().ReloadModules()

        if (runTraceMalloc): 
            checkMemoryLeak.SaveSnapshot(key, tracemalloc.take_snapshot())
        
        print("")
        print(f" --- Experiment iteration '{n + 1}' done! --- ")
        print("")
       
    if (runTraceMalloc): 
        checkMemoryLeak.WriteToFile(key, tracemalloc.get_traced_memory())

    print("")
    print(f" --- Experiment '{key}' done! --- ")
    print("")

if (runTraceMalloc): 
    checkMemoryLeak.CompareSnapshots()

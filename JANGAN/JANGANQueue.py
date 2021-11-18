from os import path, write
import traceback
import tracemalloc
import os
import time
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

tracemalloc.start()
tracemallocPath = '../../Data/TraceMalloc'

if not os.path.isdir(tracemallocPath):
    os.makedirs(tracemallocPath)

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m-%d-%Y, %H:%M:%S", named_tuple)

tracemallocFile = '/' + time_string + ' tracemalloc.txt'

file = open(tracemallocPath + tracemallocFile,"x")
file.close()

snapshot = {}
saveKeys = []

for key in expDict:
    saveKeys.append(key)
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
            print("      STACKTRACE")
            print(traceback.format_exc())
            print("")

        JANGANModuleReloader.JANGANModuleReloader().ReloadModules()

        snapshot[key] = tracemalloc.take_snapshot()
        top_stats = snapshot[key].statistics('lineno')
        
        print("")
        print(f" --- Experiment iteration '{n + 1}' done! --- ")
        print("")
    
    tm = open(tracemallocPath + tracemallocFile, 'a')
    tm.write(f"[Writing tracemalloc for {key}]")
    tm.write('\n')
    tm.close()

    for stat in top_stats[:25]:
        text = str(stat)
        with open(tracemallocPath + tracemallocFile, 'a') as f:
            f.writelines(text)
            f.write('\n')
    
    f.close()

    tm = open(tracemallocPath + tracemallocFile, 'a')
    tm.write('\n')
    tm.write('Traced memory is (current, peak):' + str(tracemalloc.get_traced_memory()))
    tm.write('\n\n')
    tm.close()
    
    print("Tracemallock is done")

    print("")
    print(f" --- Experiment '{key}' done! --- ")
    print("")

if (len(saveKeys) > 1):
    for x in range(len(saveKeys)):

        top_stats = snapshot[saveKeys].compare_to(snapshot[saveKeys], 'lineno')
        print("[ Top 10 differences ]")
        for stat in top_stats[:10]:
            print(stat)
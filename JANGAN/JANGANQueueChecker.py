from ProjectTools import ConfigHelper  

import importlib
import os

class JANGANQueueChecker():
    __cfg = None

    def __init__(self, config):
        self.__cfg = config

    def CheckConfig(self):
        expDict = self.__cfg.GetJsonValue("EXPERIMENTS","ExperimentList");
        for key in expDict:
            moduleName = expDict[key]['ModuleName']
            count = expDict[key]['AmountOfTimesToRun']
            configFile = expDict[key]['ConfigFile']

            doesModuleExist = importlib.util.find_spec(moduleName)

            if doesModuleExist is None:
                raise QueueModuleNameNotFound(f"[Experiment: {key}] Error, module '{moduleName}' from the queue not found!")

            if not os.path.exists(configFile):
                raise QueueConfigNotFound(f"[Experiment: {key}] Error, config file '{configFile}' from the queue not found!")

            if not isinstance(count, int):
                raise IterationCountIsNotANumber(f"[Experiment: {key}] Error, number of times to run is not a number!")

class QueueModuleNameNotFound(Exception):
    pass

class QueueConfigNotFound(Exception):
    pass

class IterationCountIsNotANumber(Exception):
    pass
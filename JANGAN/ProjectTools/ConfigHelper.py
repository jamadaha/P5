from ProjectTools import AutoPackageInstaller as ap
ap.CheckAndInstall("configparser")
ap.CheckAndInstall("json")

import configparser
import os
import json

class ConfigHelper():
    __config = None
    ConfigDir = ""
    ConfigOverrideDir = ""

    def __init__(self, configDir = "config.ini", configOverrideDir = "override-config.ini"):
        self.ConfigDir = configDir;
        self.ConfigOverrideDir = configOverrideDir

    def LoadConfig(self):
        self.__config = configparser.ConfigParser()
        if not os.path.exists(self.ConfigDir):
            raise Exception(f"'{os.path.abspath(os.getcwd())}\{self.ConfigDir}' not found!")
        if os.path.exists(self.ConfigOverrideDir):
            self.__config.read([self.ConfigDir, self.ConfigOverrideDir])
        else:
            self.__config.read(self.ConfigDir)

    def GetIntValue(self, category, key):
        return int(self.__config[category][key])

    def GetStringValue(self, category, key):
        return self.__config[category][key].strip('"')

    def GetJsonValue(self, category, key):
        return json.loads(self.__config[category][key].strip('"'))

    def CategoryKeyCount(self, category):
        return len(self.__config[category])
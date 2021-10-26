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
            raise ConfigFileNotFoundException(f"'{os.path.abspath(os.getcwd())}\{self.ConfigDir}' not found!")
        if os.path.exists(self.ConfigOverrideDir):
            self.__config.read([self.ConfigDir, self.ConfigOverrideDir])
        else:
            self.__config.read(self.ConfigDir)

    def CheckIfCategoryExists(self, category):
        if not self.__config.has_section(category):
            raise CategoryNotFoundException(f"Error! Category '{category}' not found in the config file!")

    def CheckIfKeyExists(self, category, key):
        self.CheckIfCategoryExists(category)
        if not self.__config.has_option(category, key):
            raise KeyNotFoundException(f"Error! Key '{key}' not found in the category '{category}' from the config file!")

    def GetIntValue(self, category, key):
        self.CheckIfKeyExists(category,key)
        return int(self.__config[category][key])

    def GetStringValue(self, category, key):
        self.CheckIfKeyExists(category,key)
        return self.__config[category][key].strip('"')

    def GetJsonValue(self, category, key):
        self.CheckIfKeyExists(category,key)
        return json.loads(self.__config[category][key].strip('"'))

    def CategoryKeyCount(self, category):
        self.CheckIfCategoryExists(category)
        return len(self.__config[category])

class CategoryNotFoundException(Exception):
    pass

class KeyNotFoundException(Exception):
    pass

class ConfigFileNotFoundException(Exception):
    pass
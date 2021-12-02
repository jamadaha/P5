from ProjectTools import AutoPackageInstaller as ap
ap.CheckAndInstall("configparser")
ap.CheckAndInstall("json")

import configparser
import os
import json
import time
import shutil

class ConfigHelper():
    __config = None
    ConfigPath = ""
    ConfigOverridePath = ""
    ConfigFileName = ""
    TokenReplacements = [("{TIMESTAMP}", time.strftime("%Y%m%d-%H%M%S"))]

    def __init__(self, configPath = "config.ini", configOverridePath = "override-config.ini", tokenReplacements = None):
        self.ConfigPath = configPath;
        self.ConfigOverridePath = configOverridePath
        if tokenReplacements != None:
            self.TokenReplacements = tokenReplacements

    def UpdateTokenReplacements(self, tokenReplacements):
        self.TokenReplacements = tokenReplacements

    def LoadConfig(self):
        self.__config = configparser.ConfigParser()
        if not os.path.exists(self.ConfigPath):
            raise ConfigFileNotFoundException(f"path: '{os.path.abspath(os.getcwd())}', file: '{self.ConfigPath}' not found!")
        if os.path.exists(self.ConfigOverridePath):
            self.__config.read([self.ConfigPath, self.ConfigOverridePath])
        else:
            self.__config.read(self.ConfigPath)

    def CopyConfigToPath(self, path):
        os.makedirs(path)
        shutil.copy(self.ConfigPath, path)

    def CheckIfCategoryExists(self, category):
        if not self.__config.has_section(category):
            raise CategoryNotFoundException(f"Error! Category '{category}' not found in the config file!")

    def CheckIfKeyExists(self, category, key):
        self.CheckIfCategoryExists(category)
        if not self.__config.has_option(category, key):
            raise KeyNotFoundException(f"Error! Key '{key}' not found in the category '{category}' from the config file!")

    def GetIntValue(self, category, key):
        self.CheckIfKeyExists(category,key)
        return self.__config.getint(category, key)

    def GetBoolValue(self, category, key):
        self.CheckIfKeyExists(category,key)
        return self.__config.getboolean(category, key)

    def GetStringValue(self, category, key):
        self.CheckIfKeyExists(category,key)
        value = self.__config[category][key]
        value = value.strip('"')
        for token in self.TokenReplacements:
            value = value.replace(token[0], token[1])
        return value

    def GetJsonValue(self, category, key):
        self.CheckIfKeyExists(category,key)
        return json.loads(self.__config[category][key].strip('"'))

    def GetFloatValue(self, category, key):
        self.CheckIfKeyExists(category, key)
        return self.__config.getfloat(category, key)

    def GetListValue(self, category, key):
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
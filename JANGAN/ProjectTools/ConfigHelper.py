from ProjectTools import AutoPackageInstaller as ap
ap.CheckAndInstall("configparser")
ap.CheckAndInstall("json")

import configparser
import os
import json

__config = configparser.ConfigParser()
if not os.path.exists('config.ini'):
    raise Exception('config.ini not found!', os.path)
if os.path.exists('override-config.ini'):
    __config.read(['config.ini','override-config.ini'])
else:
    __config.read('config.ini')

def GetIntValue(category, key):
    return int(__config[category][key])

def GetStringValue(category, key):
    return __config[category][key].strip('"')

def GetJsonValue(category, key):
    return json.loads(__config[category][key].strip('"'))

def CategoryKeyCount(category):
    return len(__config[category])
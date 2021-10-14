import configparser
import os
import json

__config = configparser.ConfigParser()
if os.path.exists('override-config.ini'):
    __config.read(['config.ini','override-config.ini'])
else:
    __config.read('config.ini')

def GetStringValue(category, key):
    return __config[category][key].strip('"')

def GetJsonValue(category, key):
    return json.loads(__config[category][key].strip('"'))
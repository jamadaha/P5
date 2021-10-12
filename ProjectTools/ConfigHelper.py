import configparser
import os

config = configparser.ConfigParser()
if os.path.exists('override-config.ini'):
    config.read(['config.ini','override-config.ini'])
else:
    config.read('config.ini')
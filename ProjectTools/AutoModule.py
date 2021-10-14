import os

def CheckAndInstall(packageName):
    try:
        return __import__(packageName)
    except ImportError:
        os.system("pip install "+ packageName)

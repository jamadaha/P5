import sys
import subprocess

def CheckAndInstall(packageName):
    try:
        return __import__(packageName)
    except ImportError:
        while True:
            Question = input("Package '" + packageName + "' is missing. Wanna install it? (y/n)")
            if Question == "y":
                subprocess.check_call([sys.executable, "-m", "pip", "install", packageName])
                break
            elif Question == "n":
                print("Exiting application.")
                sys.exit()

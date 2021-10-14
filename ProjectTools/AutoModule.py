import sys
import subprocess
import os

os.environ["AutoModule_YesToAllModules"] = "False"

def CheckAndInstall(packageName):
    try:
        return __import__(packageName)
    except ImportError:
        if os.environ["AutoModule_YesToAllModules"] == "True":
            print(" --- Installing package " + packageName + " ---")
            subprocess.check_call([sys.executable, "-m", "pip", "install", packageName])
        else:
            while True:
                Question = input("Package '" + packageName + "' is missing. Wanna install it? (y/n)(type Y to say yes to all):")
                if Question == "Y":
                    os.environ["AutoModule_YesToAllModules"] = "True";
                    CheckAndInstall(packageName)
                    break
                elif Question == "y":
                    print(" --- Installing package " + packageName + " ---")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", packageName])
                    break
                elif Question == "n":
                    print("Exiting application.")
                    sys.exit()

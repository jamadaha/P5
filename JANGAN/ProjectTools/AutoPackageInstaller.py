import sys
import subprocess
import os
import importlib

def CheckAndInstall(packageName, installName = None):
    spam_loader = importlib.find_loader(packageName)
    found = spam_loader is not None

    if not found:
        if not installName:
            installName = packageName

        if not os.getenv("AutoPackageInstaller_YesToAllModules"):
            os.environ["AutoPackageInstaller_YesToAllModules"] = "False"

        if os.environ["AutoPackageInstaller_YesToAllModules"] == "True":
            print(" --- Installing package " + installName + " ---")
            subprocess.check_call([sys.executable, "-m", "pip", "install", installName])
        else:
            while True:
                Question = input("Package '" + installName + "' is missing. Wanna install it? (y/n)(type Y to say yes to all):")
                if Question == "Y":
                    os.environ["AutoPackageInstaller_YesToAllModules"] = "True"
                    CheckAndInstall(packageName)
                    break
                elif Question == "y":
                    print(" --- Installing package " + installName + " ---")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", installName])
                    break
                elif Question == "n":
                    print("Exiting application.")
                    sys.exit()

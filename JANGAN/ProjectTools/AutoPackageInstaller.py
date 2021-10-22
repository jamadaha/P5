import sys
import subprocess
import os

def CheckAndInstall(packageName, installName = None):
    try:
        return __import__(packageName)
    except ImportError:
        if not installName:
            installName = packageName

        if not os.environ["AutoPackageInstaller_YesToAllModules"]:
            os.environ["AutoPackageInstaller_YesToAllModules"] = "False"

        if os.environ["AutoPackageInstaller_YesToAllModules"] == "True":
            print(" --- Installing package " + installName + " ---")
            subprocess.check_call([sys.executable, "-m", "pip", "install", installName])
        else:
            while True:
                Question = input("Package '" + installName + "' is missing. Wanna install it? (y/n)(type Y to say yes to all):")
                if Question == "Y":
                    os.environ["AutoPackageInstaller_YesToAllModules"] = "True";
                    CheckAndInstall(packageName)
                    break
                elif Question == "y":
                    print(" --- Installing package " + installName + " ---")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", installName])
                    break
                elif Question == "n":
                    print("Exiting application.")
                    sys.exit()

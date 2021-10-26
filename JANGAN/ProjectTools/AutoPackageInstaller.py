import sys
import subprocess
import os

def CheckAndInstall(packageName, installName = None):
    try:
        return __import__(packageName)
    except ImportError:
        if not installName:
            installName = packageName

        if not os.getenv("AutoPackageInstaller_YesToAllModules"):
            os.environ["AutoPackageInstaller_YesToAllModules"] = "False"

        if os.environ["AutoPackageInstaller_YesToAllModules"] == "True":
            __InstallPackage(installName)
        else:
            while True:
                Question = input("Package '" + installName + "' is missing. Wanna install it? (y/n)(type Y to say yes to all):")
                if Question == "Y":
                    os.environ["AutoPackageInstaller_YesToAllModules"] = "True";
                    CheckAndInstall(installName)
                    break
                elif Question == "y":
                    __InstallPackage(installName)
                    break
                elif Question == "n":
                    break

def __InstallPackage(packageName):
    print(f" --- Installing package '{packageName}' ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", packageName])
    print(f" --- Package '{packageName}' installed! ---")
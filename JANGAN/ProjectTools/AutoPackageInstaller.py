import sys
import subprocess
import os

AutoUpdate = False
InstallAllMissingModules = False
UpdateAllModules = False

__CheckedModules_Updates = {}
__CheckedModules_Installed = {}

def CheckAndInstall(packageName, installName = None):
    __CheckForUpdate("pip")

    try:
        if not packageName in __CheckedModules_Installed:
            __CheckedModules_Installed[packageName] = True
            print(f"Checking if package '{packageName}' is installed: ", end='')
            mod = __import__(packageName)
            print("GOOD")
            if AutoUpdate == True:
                __CheckForUpdate(mod.__package__)
    except ImportError:
        print("ERROR")

        if not installName:
            installName = packageName

        if InstallAllMissingModules == True:
            __InstallPackage(installName)
        else:
            while True:
                Question = input("Package '" + installName + "' is missing. Wanna install it? (y/n)(type Y to say yes to all):")
                if Question == "Y":
                    InstallAllMissingModules = True
                    __InstallPackage(installName)
                    break
                elif Question == "y":
                    __InstallPackage(installName)
                    break
                elif Question == "n":
                    break

def __CheckForUpdate(packageName):
    if packageName != "":
        if not packageName in __CheckedModules_Updates:
            __CheckedModules_Updates[packageName] = True
            print(f"Checking package '{packageName}' for updates: ", end='')

            current_version = __GetCurrentPackageVersion(packageName)
            latest_version = __GetLatestPackageVersion(packageName)

            if latest_version != current_version:
                print("WARN")
                print(f"Warning! A newer version of the '{packageName}' is available! Installed: {current_version}, latest: {latest_version}")
                if UpdateAllModules == True:
                    __UpdatePackage(packageName)
                else:
                    while True:
                        Question = input("Package '" + packageName + "' can be updated. Wanna update it? (y/n)(type Y to say yes to all):")
                        if Question == "Y":
                            UpdateAllModules = True
                            __UpdatePackage(packageName)
                            break
                        elif Question == "y":
                            __UpdatePackage(packageName)
                            break
                        elif Question == "n":
                            break
            else:
                print("GOOD")

def __InstallPackage(packageName):
    print(f" --- Installing package '{packageName}' ---")
    subprocess.run(["py", "-m", "pip", "install", packageName])
    print(f" --- Package '{packageName}' installed! ---")


def __UpdatePackage(packageName):
    print(f" --- Updating package '{packageName}' ---")
    subprocess.run(["py", "-m", "pip", "install", packageName, "--upgrade"])
    print(f" --- Package '{packageName}' updated! ---")

def __GetCurrentPackageVersion(packageName):
    current_version = str(subprocess.run([sys.executable, '-m', 'pip', 'show', '{}'.format(packageName)], capture_output=True, text=True))
    if current_version.count("WARNING") > 0:
        return "none"
    current_version = current_version[current_version.find('Version:')+8:]
    current_version = current_version[:current_version.find('\\n')].replace(' ','') 

    return current_version;

def __GetLatestPackageVersion(packageName):
    latest_version = str(subprocess.run([sys.executable, '-m', 'pip', 'install', '{}==random'.format(packageName)], capture_output=True, text=True))
    if latest_version.count("WARNING") > 0:
        return "none"
    latest_version = latest_version[latest_version.find('(from versions:')+15:]
    latest_version = latest_version[:latest_version.find(')')]
    latest_version = latest_version.replace(' ','').split(',')[-1]

    return latest_version;
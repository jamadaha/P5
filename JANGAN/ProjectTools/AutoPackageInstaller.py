import sys
import subprocess
import os

def CheckAndInstall(packageName, installName = None):
    if not os.getenv("AutoPackageInstaller_AutoUpdate"):
        os.environ["AutoPackageInstaller_AutoUpdate"] = "False"

    try:
        mod = __import__(packageName)
        if os.getenv("AutoPackageInstaller_AutoUpdate") == "True":
            if mod.__package__ != "":
                if not os.getenv(f"AutoPackageInstaller_HaveCheckedForUpdates_{packageName}"):
                    print(f"Checking package {mod.__package__} for updates...")
                    CheckForUpdate(mod.__package__)
                    os.environ[f"AutoPackageInstaller_HaveCheckedForUpdates_{packageName}"] = "True"
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
                    __InstallPackage(installName)
                    break
                elif Question == "y":
                    __InstallPackage(installName)
                    break
                elif Question == "n":
                    break

def CheckForUpdate(packageName):
    current_version = __GetCurrentPackageVersion(packageName)
    latest_version = __GetLatestPackageVersion(packageName)

    if latest_version != current_version:
        print(f"Warning! A newer version of the '{packageName}' is available! Installed: {current_version}, latest: {latest_version}")
        if not os.getenv("AutoPackageInstaller_YesToAllUpdates"):
            os.environ["AutoPackageInstaller_YesToAllUpdates"] = "False"

        if os.environ["AutoPackageInstaller_YesToAllUpdates"] == "True":
            __UpdatePackage(packageName)
        else:
            while True:
                Question = input("Package '" + packageName + "' can be updated. Wanna update it? (y/n)(type Y to say yes to all):")
                if Question == "Y":
                    os.environ["AutoPackageInstaller_YesToAllUpdates"] = "True";
                    __UpdatePackage(packageName)
                    break
                elif Question == "y":
                    __UpdatePackage(packageName)
                    break
                elif Question == "n":
                    break

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
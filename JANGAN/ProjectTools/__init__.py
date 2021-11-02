import os
from ProjectTools import AutoPackageInstaller as ap
if os.getenv("AutoPackageInstaller_YesToAllModules"):
    ap.InstallAllMissingModules = True

import unittest
from unittest.mock import patch
import os

class AutoPackageInstallerTests(unittest.TestCase):
    def test_CheckAndInstall_SetsEnvironmentVariable_IfNo(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap

        # ACT
        with patch('builtins.input', return_value = 'n'):
            ap.CheckAndInstall("nonexistentpackage")

        # ASSERT
        self.assertEqual("False", os.getenv("AutoPackageInstaller_YesToAllModules"))

        CleanupEnvironment();

    def test_CheckAndInstall_SetsEnvironmentVariable_IfLargeYes(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap

        # ACT
        with patch.object(ap, '__InstallPackage', return_value = None) as insPackObj:
            with patch('builtins.input', return_value = 'Y'):
                ap.CheckAndInstall("nonexistentpackage")

        # ASSERT
        self.assertEqual("True", os.getenv("AutoPackageInstaller_YesToAllModules"))

        CleanupEnvironment();

    def test_CheckAndInstall_DontSetEnvironmentVariable_IfSmallYes(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap

        # ACT
        with patch.object(ap, '__InstallPackage', return_value = None) as insPackObj:
            with patch('builtins.input', return_value = 'y'):
                ap.CheckAndInstall("nonexistentpackage")

        # ASSERT
        self.assertEqual("False", os.getenv("AutoPackageInstaller_YesToAllModules"))

        CleanupEnvironment();

def CleanupEnvironment():
    if os.getenv("AutoPackageInstaller_YesToAllModules"):
        del os.environ["AutoPackageInstaller_YesToAllModules"]
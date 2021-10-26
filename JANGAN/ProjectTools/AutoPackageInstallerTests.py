import unittest
from unittest.mock import patch
import os

class AutoPackageInstallerTests(unittest.TestCase):
    @patch('ProjectTools.AutoPackageInstaller.GetInput', return_value='n')
    def test_SetsEnvironmentVariable_IfNo(self, input):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap

        # ACT
        ap.CheckAndInstall("nonexistentpackage")

        # ASSERT
        self.assertEqual("False",os.getenv("AutoPackageInstaller_YesToAllModules"))

    @patch('ProjectTools.AutoPackageInstaller.GetInput', return_value='Y')
    @patch('ProjectTools.AutoPackageInstaller.InstallPackage', return_value='')
    def test_SetsEnvironmentVariable_IfLargeYes(self, input, package):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap

        # ACT
        ap.CheckAndInstall("nonexistentpackage")

        # ASSERT
        self.assertEqual("True",os.getenv("AutoPackageInstaller_YesToAllModules"))

    @patch('ProjectTools.AutoPackageInstaller.GetInput', return_value='y')
    @patch('ProjectTools.AutoPackageInstaller.InstallPackage', return_value='')
    def test_DontSetEnvironmentVariable_IfSmallYes(self, input, package):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap

        # ACT
        ap.CheckAndInstall("nonexistentpackage")

        # ASSERT
        self.assertEqual("False",os.getenv("AutoPackageInstaller_YesToAllModules"))

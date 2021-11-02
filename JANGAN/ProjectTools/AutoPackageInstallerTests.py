import unittest
from unittest.mock import patch
import os
from importlib import reload

class AutoPackageInstallerTests(unittest.TestCase):

    # Initialization of Module
    #region Initialization of Module
    def test_SetsDefaultOf_AutoUpdate(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap

        # ACT
        # ASSERT
        self.assertEqual(False, ap.AutoUpdate)

    def test_SetsDefaultOf_InstallAllMissingModules(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap

        # ACT
        # ASSERT
        self.assertEqual("Ask", ap.InstallAllMissingModules)

    def test_SetsDefaultOf_UpdateAllModules(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap

        # ACT
        # ASSERT
        self.assertEqual("Ask", ap.UpdateAllModules)
    #endregion

    # InstallAllMissingModules Tests
    #region

    def test_CheckAndInstall_IfAsk_SetsInstallAllMissingModulesToTrueIfY(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap
        reload(ap)

        # ACT
        with patch.object(ap, '__InstallPackage', return_value = None) as insPackObj:
            with patch('builtins.input', return_value = 'Y'):
                ap.CheckAndInstall("nonexistentpackage")

        # ASSERT
        self.assertEqual("True", ap.InstallAllMissingModules)

    def test_CheckAndInstall_IfAsk_SetsInstallAllMissingModulesToFalseIfN(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap
        reload(ap)

        # ACT
        with patch('builtins.input', return_value = 'N'):
            ap.CheckAndInstall("nonexistentpackage")

        # ASSERT
        self.assertEqual("False", ap.InstallAllMissingModules)

    def test_CheckAndInstall_IfAsk_SetsInstallAllMissingModulesUnchangedIfSmallY(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap
        reload(ap)

        # ACT
        with patch.object(ap, '__InstallPackage', return_value = None) as insPackObj:
            with patch('builtins.input', return_value = 'y'):
                ap.CheckAndInstall("nonexistentpackage")

        # ASSERT
        self.assertEqual("Ask", ap.InstallAllMissingModules)

    def test_CheckAndInstall_IfAsk_SetsInstallAllMissingModulesUnchangedIfSmallN(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap
        reload(ap)

        # ACT
        with patch('builtins.input', return_value = 'n'):
            ap.CheckAndInstall("nonexistentpackage")

        # ASSERT
        self.assertEqual("Ask", ap.InstallAllMissingModules)
    #endregion

    # UpdateAllModules Tests
    #region

    def test_CheckAndInstall_IfAsk_SetsUpdateAllModulesToTrueIfY(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap
        reload(ap)

        # ACT
        with patch.object(ap, '__UpdatePackage', return_value = None):
            with patch.object(ap, '__GetCurrentPackageVersion', return_value = "1"):
                with patch.object(ap, '__GetLatestPackageVersion', return_value = "2"):
                    with patch('builtins.input', return_value = 'Y'):
                        ap.CheckForUpdate("nonexistentpackage")

        # ASSERT
        self.assertEqual("True", ap.UpdateAllModules)

    def test_CheckAndInstall_IfAsk_SetsUpdateAllModulesToFalseIfN(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap
        reload(ap)

        # ACT
        with patch.object(ap, '__UpdatePackage', return_value = None):
            with patch.object(ap, '__GetCurrentPackageVersion', return_value = "1"):
                with patch.object(ap, '__GetLatestPackageVersion', return_value = "2"):
                    with patch('builtins.input', return_value = 'N'):
                        ap.CheckForUpdate("nonexistentpackage")

        # ASSERT
        self.assertEqual("False", ap.UpdateAllModules)

    def test_CheckAndInstall_IfAsk_SetsUpdateAllModulesUnchangedIfSmallY(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap
        reload(ap)

        # ACT
        with patch.object(ap, '__UpdatePackage', return_value = None):
            with patch.object(ap, '__GetCurrentPackageVersion', return_value = "1"):
                with patch.object(ap, '__GetLatestPackageVersion', return_value = "2"):
                    with patch('builtins.input', return_value = 'y'):
                        ap.CheckForUpdate("nonexistentpackage")

        # ASSERT
        self.assertEqual("Ask", ap.UpdateAllModules)

    def test_CheckAndInstall_IfAsk_SetsUpdateAllModulesUnchangedIfSmallN(self):
        # ARRANGE
        from ProjectTools import AutoPackageInstaller as ap
        reload(ap)

        # ACT
        with patch.object(ap, '__UpdatePackage', return_value = None):
            with patch.object(ap, '__GetCurrentPackageVersion', return_value = "1"):
                with patch.object(ap, '__GetLatestPackageVersion', return_value = "2"):
                    with patch('builtins.input', return_value = 'n'):
                        ap.CheckForUpdate("nonexistentpackage")

        # ASSERT
        self.assertEqual("Ask", ap.UpdateAllModules)
    #endregion

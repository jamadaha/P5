import unittest

class ConfigHelperTests(unittest.TestCase):
    # Config Loading
    #region Config Loading
    def test_CanLoadConfig(self):
        # ARRANGE
        from ProjectTools import ConfigHelper
        cfg = ConfigHelper.ConfigHelper("ProjectTools/TestFiles/test-config.ini")

        # ACT
        cfg.LoadConfig()
        length : int = cfg.CategoryKeyCount("TESTCATEGORY1")

        # ASSERT
        self.assertLess(0, length)

    def test_ThrowsIfConfigFileNotFound(self):
        # ARRANGE
        from ProjectTools import ConfigHelper
        cfg = ConfigHelper.ConfigHelper("nonexistent-config.ini")

        # ACT
        with self.assertRaises(ConfigHelper.ConfigFileNotFoundException):
            cfg.LoadConfig()

    #endregion

    # Key/Category checking
    #region Key/Category checking

    def test_CanGetAmountOfKeysInCategory(self):
        # ARRANGE
        from ProjectTools import ConfigHelper
        cfg = ConfigHelper.ConfigHelper("ProjectTools/TestFiles/test-config.ini")
        cfg.LoadConfig()

        # ACT
        length : int = cfg.CategoryKeyCount("CATEGORY_WITH_3_ITEMS")

        # ASSERT
        self.assertEqual(3, length)

    def test_ThrowsIfCategoryNotFound(self):
        # ARRANGE
        from ProjectTools import ConfigHelper
        cfg = ConfigHelper.ConfigHelper("ProjectTools/TestFiles/test-config.ini")
        cfg.LoadConfig()

        # ACT
        with self.assertRaises(ConfigHelper.CategoryNotFoundException):
            cfg.CheckIfCategoryExists("NONEXISTINGCATEGORY")

    def test_ThrowsIfKeyNotFound(self):
        # ARRANGE
        from ProjectTools import ConfigHelper
        cfg = ConfigHelper.ConfigHelper("ProjectTools/TestFiles/test-config.ini")
        cfg.LoadConfig()

        # ACT
        with self.assertRaises(ConfigHelper.KeyNotFoundException):
            cfg.CheckIfKeyExists("TESTCATEGORY1", "NonExistingKey")

    #endregion

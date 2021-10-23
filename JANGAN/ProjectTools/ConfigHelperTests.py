import unittest

class ConfigHelperTests(unittest.TestCase):
    def test_CanLoadConfig(self):
        # ARRANGE
        from ProjectTools import ConfigHelper
        cfg = ConfigHelper.ConfigHelper("config.ini")

        # ACT
        cfg.LoadConfig()
        length : int = cfg.CategoryKeyCount("DATAGENERATOR")

        # ASSERT
        self.assertLess(0, length)


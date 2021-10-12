import unittest

class ConfigHelperTests(unittest.TestCase):
    def test_CanLoadConfig(self):
        # ARRANGE
        from ProjectTools import ConfigHelper as cfg

        # ACT
        length : int = len(cfg.config['DATAGENERATOR'])

        # ASSERT
        self.assertLess(0, length)


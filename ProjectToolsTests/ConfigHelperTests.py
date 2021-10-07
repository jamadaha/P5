import unittest

class ConfigHelperTests(unittest.TestCase):
    def test_CanLoadConfig(self):
        # ARRANGE
        import ProjectTools.ConfigHelper as cfg
        
        # ACT
        length = len(cfg.config['DATAGENERATOR'])

        # ASSERT
        self.assertLess(0, length)


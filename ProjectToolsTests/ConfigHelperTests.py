import unittest

class ConfigHelperTests(unittest.TestCase):

    def test_LoadedConfigCantLoad(self):
        # ARRANGE
        import ProjectTools.ConfigHelper as cfg
        
        # ACT
        length = len(cfg.config['DATAGENERATOR'])

        # ASSERT
        self.assertLessEqual(0, length)


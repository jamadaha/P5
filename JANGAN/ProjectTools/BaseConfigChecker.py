from ProjectTools import ConfigHelper as cfgh

class BaseConfigChecker():
    ThrowIfConfigFileBad = True
    CheckedKeyCount = 0
    cfg = None

    def __init__(self, cfg : cfgh.ConfigHelper, throwIfConfigFileBad):
        self.cfg = cfg
        self.ThrowIfConfigFileBad = throwIfConfigFileBad

    def CheckConfig(self):
        raise Exception("CheckConfig(...) not implemented!")

    def CheckKey(self, category, key):
        if self.ThrowIfConfigFileBad == True:
            self.cfg.CheckIfKeyExists(category, key)
        else:
            try:
                self.cfg.CheckIfKeyExists(category, key)
            except (cfgh.CategoryNotFoundException, cfgh.KeyNotFoundException) as e:
                print(f"Warning! A config key/category is missing! (category: {category}, key: {key})")
        self.CheckedKeyCount += 1

    def CheckKeyCount(self, category):
        keyCount = self.cfg.CategoryKeyCount(category)
        if keyCount != self.CheckedKeyCount:
            if self.ThrowIfConfigFileBad == True:
                raise Exception(f"Error! Config category '{category}' key count did not match the expected! (expected: {keyCount}, actual {self.CheckedKeyCount})")
            else:
                print(f"Warning! Config category '{category}' key count did not match the expected! (expected: {keyCount}, actual {self.CheckedKeyCount})")
from ProjectTools import ConfigHelper as cfgh

class JANGANConfigChecker():
    ThrowIfConfigFileBad = True

    def CheckConfig(self, cfg : cfgh.ConfigHelper, throwIfConfigFileBad):
        self.ThrowIfConfigFileBad = throwIfConfigFileBad
        print(" --- Checking config file integrity --- ")

        self.__CheckKey(cfg, "DATAGENERATOR","BasePath")

        print(" --- Done! --- ")

    def __CheckKey(self, cfg : cfgh.ConfigHelper, category, key):
        if self.ThrowIfConfigFileBad == True:
            cfg.CheckIfKeyExists(category, key)
        else:
            try:
                cfg.CheckIfKeyExists(category, key)
            except (CategoryNotFoundException, KeyNotFoundException) as e:
                print(f"Warning! A config key/category is missing! (category: {category}, key: {key})")
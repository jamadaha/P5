from importlib import reload

class JANGANModuleReloader():

    def ReloadModules(self):
        print(" --- Reloading modules --- ")

        import CGAN as cg
        reload(cg.CGANKerasModel)
        reload(cg.CGANTrainer)
        reload(cg.DatasetFormatter)
        reload(cg.DatasetLoader)
        reload(cg.LayerDefinition)
        reload(cg.LetterProducer)
        reload(cg)

        import DataGenerator as dg
        reload(dg.DataExtractor)
        reload(dg.FileImporter)
        reload(dg.SharedFunctions)
        reload(dg.TextSequence)
        reload(dg)

        import JANGAN as jg
        reload(jg)

        print(" --- Done! --- ")

from importlib import reload

class JANGANModuleReloader():

    def ReloadModules(self):
        print(" --- Reloading modules --- ")

        import CGAN as cg
        reload(cg.CGANKerasModel)
        reload(cg.CGANTrainer)
        reload(cg.LayerDefinition)
        reload(cg.LetterProducer)
        reload(cg)

        import DataGenerator as dg
        reload(dg.DataExtractor)
        reload(dg.SharedFunctions)
        reload(dg.TextSequence)
        reload(dg.FileImporter)
        reload(dg)

        import Classifier as cf
        reload(cf.ClassifierKerasModel)
        reload(cf.ClassifierTrainer)
        reload(cf.LayerDefinition)
        reload(cf)

        import DatasetLoader as dl
        reload(dl.DatasetFormatter)
        reload(dl.DatasetLoader)
        reload(dl.DiskReader)
        reload(dl)

        import ProjectTools as pt
        reload(pt.BaseKerasModelTrainer)
        reload(pt.BaseMLModel)
        reload(pt.CSVLogger)
        reload(pt.TFLogger)
        reload(pt)

        import JANGAN as jg
        reload(jg)

        print(" --- Done! --- ")

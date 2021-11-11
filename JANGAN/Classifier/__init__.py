from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

class Classifier():
    def __init__(self):
        pass

    def TrainClassifier(self):
        # Train the classifier, or use a checkpoinht if its there
        pass

    def ProduceStatistics(self):
        # Output from the classifier. Could be accuracy or something
        pass

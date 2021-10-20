import AutoPackageInstaller as ap

class TextSequence:
    TextPath = ""
    TextFileQueue = []
    TextFileStream = []

    def __init__(self, textPath):
        self.TextPath = textPath
        self.LoadTextStream()

    def LoadTextStream(self):
        import os
        print("Loading texts")
        self.TextFileQueue = os.listdir(self.TextPath)
        if not self.TextFileQueue:
            raise Exception("Text files not found!")
        self.TextFileStream = open(
            self.TextPath + self.TextFileQueue[0], 'r', encoding='utf-8')
        self.TextFileQueue.pop(0)

    def GetNextLetter(self):
        return self.TextFileStream.read(1)

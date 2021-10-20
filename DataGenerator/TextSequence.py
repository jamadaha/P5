import AutoPackageInstaller as ap

class TextSequence:
    TextPath = ""
    TextFileQueue = []
    TextFileStream = []

    def __init__(self, textPath):
        self.TextPath = textPath
        self.ImportTexts()
        self.LoadNextStream()

    def ImportTexts(self):
        import os
        print("Loading texts")
        self.TextFileQueue = os.listdir(self.TextPath)
        if not self.TextFileQueue:
            raise Exception("Text files not found!")
        

    def LoadNextStream(self):
        self.TextFileStream = open(
            self.TextPath + self.TextFileQueue[0], 'r', encoding='utf-8')
        self.TextFileQueue.pop(0)

    def GetNextLetter(self):
        char = self.TextFileStream.read(1)
        if len(char) == 0:
            if len(self.TextFileQueue) == 0:
                raise Exception("Ran out of text files, please add more")
            self.LoadNextStream()
        return char

class FileImporter:
    TextPath = ""
    TextDownloadURL = []
    LetterDownloadURL = ""
    TempDownloadLetterPath = ""
    TempDownloadLetterFileName = ""

    def __init__(self, textPath, letterDownloadURL, textDownloadURL, tempDownloadLetterPath, tempDownloadLetterFileName):
        self.TextPath = textPath
        self.LetterDownloadURL = letterDownloadURL
        self.TextDownloadURL = textDownloadURL
        self.TempDownloadLetterPath = tempDownloadLetterPath
        self.TempDownloadLetterFileName = tempDownloadLetterFileName

    def ImportFiles(self):
        print("Importing files")
        self.ImportTexts()
        self.ImportLetters()

    def ImportTexts(self):
        from . import SharedFunctions as sf
        print("Importing texts")
        for file in self.TextDownloadURL:
            sf.DownloadIfNotExist(
                self.TextDownloadURL[file], self.TextPath, file + '.txt')

    def ImportLetters(self):
        from . import SharedFunctions as sf
        print("Importing data")
        return sf.DownloadIfNotExist(
            self.LetterDownloadURL, self.TempDownloadLetterPath, self.TempDownloadLetterFileName)

import AutoPackageInstaller as ap

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
        from SharedFunctions import DownloadIfNotExist
        print("Importing texts")
        for file in self.TextDownloadURL:
            DownloadIfNotExist(
                self.TextDownloadURL[file], self.TextPath, file + '.txt')

    def ImportLetters(self):
        from SharedFunctions import DownloadIfNotExist
        print("Importing data")
        return DownloadIfNotExist(
            self.LetterDownloadURL, self.TempDownloadLetterPath, self.TempDownloadLetterFileName)

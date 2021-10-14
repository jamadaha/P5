from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("zipfile")
ap.CheckAndInstall("tqdm")

class FileLoader:
    import os
    import zipfile
    from tqdm import tqdm
    TextPath = ""
    LetterPath = ""
    TextDownloadURL = []
    LetterDownloadURL = ""
    TempDownloadLetterPath = ""
    TempDownloadLetterFileName = "letters.zip"
    TextFileQueue = []
    TextFileStream = []
    LetterPaths = {}

    def __init__(self, textPath, letterPath, letterDownloadURL, textDownloadURL, tempDownloadLetterPath):
        self.TextPath = textPath
        self.LetterPath = letterPath
        self.LetterDownloadURL = letterDownloadURL
        self.TextDownloadURL = textDownloadURL
        self.TempDownloadLetterPath = tempDownloadLetterPath

    def ImportAllData(self):
        self.CheckAndCreatePaths()
        self.LoadLetterPaths()
        self.GatherLetterPaths()

    def CheckAndCreatePaths(self):
        print("Checking and creating file paths ... ", end="")
        self.ImportTexts()
        zipFile = self.ImportLetters()
        if not self.os.path.isdir(self.LetterPath):
            self.ExtractLetters(zipFile)

    def ImportTexts(self):
        from SharedFunctions import DownloadIfNotExist
        for file in self.TextDownloadURL:
            DownloadIfNotExist(
                self.TextDownloadURL[file], self.TextPath, file + '.txt')

    def ImportLetters(self):
        from SharedFunctions import DownloadIfNotExist
        return DownloadIfNotExist(
            self.LetterDownloadURL, self.TempDownloadLetterPath, self.TempDownloadLetterFileName)

    def ExtractLetters(self, zipFile):
        print("Beginning extraction (This can take a bit to start) ... ")
        self.os.makedirs(self.LetterPath)
        with self.zipfile.ZipFile(zipFile, "r") as zf:
            fileList = []
            for zipFile in zf.namelist():
                if 'train' not in zipFile:
                    fileList.append(zipFile)
            for zipFile in self.tqdm(iterable=fileList, total=len(fileList)):
                zf.extract(member=zipFile, path=(self.LetterPath + "../"))
        print("Done")

    def LoadLetterPaths(self):
        print("Loading letter paths ... ", end="")
        self.TextFileQueue = self.os.listdir(self.TextPath)
        if not self.TextFileQueue:
            raise Exception("Text files not found!")
        self.TextFileStream = open(
            self.TextPath + self.TextFileQueue[0], 'r', encoding='utf-8')
        self.TextFileQueue.pop(0)
        print("Done")

    def GatherLetterPaths(self):
        print("Gathering letter paths")
        self.LetterPaths = {}
        for i in self.tqdm([*range(65, 91), *range(97, 122)]):
            self.LetterPaths[chr(i)] = {}
            hexLetter = hex(i).split('x')[-1]
            self.LetterPaths[chr(i)]['hex'] = hexLetter

        print("\nBind letter paths to a letters", end="")
        # for each letter (lowercase/uppercase)
        for n in self.tqdm(self.LetterPaths):
            self.LetterPaths[n]['paths'] = []
            # for each hsf num
            for i in range(0, 8):
                hsfPath = self.LetterPath + \
                    self.LetterPaths[n]['hex'] + "/hsf_" + str(i) + "/"
                if self.os.path.isdir(hsfPath):
                    pictures = self.os.listdir(hsfPath)
                    # for each picture in directory
                    for t in pictures:
                        self.LetterPaths[n]['paths'].append(hsfPath + t)

    def Finish(self):
        self.TextFileStream.close()

# Get 2 flows of data - Text files and pictures

# Text files
# Move .txt files into the folder DataGenerator/InputText/...
# These files should represent a valid letter sequence, not random (If true random possible, then that is fine)
# A good way to gather these would be to use non-copyrighted books

class FileLoader:
    import os
    import wget
    import zipfile
    from tqdm import tqdm
    TextPath = ""
    LetterPath = ""
    TextDownloadURL = []
    LetterDownloadURL = ""
    TempDownloadLetterPath = "./letters.zip"
    TextFileQueue = []
    TextFileStream = []
    LetterPaths = {}

    def __init__(self, textPath, letterPath, letterDownloadURL, textDownloadURL):
        self.TextPath = textPath
        self.LetterPath = letterPath
        self.LetterDownloadURL = letterDownloadURL
        for URL in textDownloadURL:
            self.TextDownloadURL.append(URL)

    def CheckAndCreatePaths(self):
        print("Checking and creating file paths ... ", end="")
        if not self.os.path.isdir(self.TextPath):
            self.ImportTexts()
        if not self.os.path.isdir(self.LetterPath):
            self.ImportLetters()

    def ImportTexts(self):
        print("Texts not found ...")
        self.os.makedirs(self.TextPath)
        for file in self.TextDownloadURL:
            print("Downloading " + file[0])
            self.wget.download(file[1], self.TextPath + file[0] + ".txt")
            print()

    def ImportLetters(self):
        print("Letters not found ... ", end="")
        file = None
        if not self.os.path.isfile(self.TempDownloadLetterPath):
            print("Letters not downloaded ... Beginning download of letters ... ")
            file = self.wget.download(
                self.LetterDownloadURL, self.TempDownloadLetterPath)
            print()
            print("Letters downloaded ... ", end="")
        else:
            file = self.TempDownloadLetterPath
            print("Letters zip file already downloaded ... ", end="")

        print("Beginning extraction ... ")
        # self.os.makedirs(self.LetterPath)
        with self.zipfile.ZipFile(file, "r") as zf:
            fileList = []
            for file in zf.namelist():
                if 'train' not in file:
                    fileList.append(file)
            for file in self.tqdm(iterable=fileList, total=len(fileList)):
                zf.extract(member=file, path=(self.LetterPath + "../"))

        # Move up one folder
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

        print("Bind letter paths to a letters")
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


class CSVGenerator:
    import csv
    FileName = ""
    Fields = ()
    CSVFile = {}
    CSVWriter = {}

    def __init__(self, fileName, fields):
        self.FileName = fileName
        self.Fields = fields
        self.CSVFile = open(self.FileName, 'wt', newline='')
        self.CSVWriter = self.csv.writer(self.CSVFile, delimiter=',')
        self.CSVWriter.writerow(fields)

    def GenerateCSVData(self, textFileStream, letterPaths, textPath, textFileQueue):
        print("Filling CSV with data ... ", end="")
        while 1:
            output = textFileStream.read(1)
            if output:
                if output in letterPaths:
                    # if no more pictures for that letter
                    if len(letterPaths[output]['paths']) == 0:
                        break
                    self.CSVWriter.writerow(
                        [output, letterPaths[output]['paths'][0]])
                    letterPaths[output]['paths'].pop(0)
            # in case that the stream reaches end of file, goto next file
            else:
                if len(textFileQueue) > 0:
                    textFileStream = open(
                        textPath + textFileQueue[0], 'r', encoding='utf-8')
                    textFileQueue.pop(0)
                else:
                    break
        print("Done")

    def Finish(self):
        self.CSVFile.close()


fl = FileLoader("./DataGenerator/InputText/", "./DataGenerator/InputLetters/by_class/", "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
                [['AChristmasCarol', 'https://www.gutenberg.org/files/46/46-0.txt'], ['PrideandPrejudice', 'https://www.gutenberg.org/files/1342/1342-0.txt']])
fl.CheckAndCreatePaths()
fl.LoadLetterPaths()
fl.GatherLetterPaths()

cg = CSVGenerator("data.csv", ('Letter', 'Path'))
cg.GenerateCSVData(fl.TextFileStream, fl.LetterPaths,
                   fl.TextPath, fl.TextFileQueue)

fl.Finish()
cg.Finish()

print("Dataset generated!")
input("Press Enter to exit...")

>>>>>> > main

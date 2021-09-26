# Get 2 flows of data - Text files and pictures

## Text files
### Move .txt files into the folder DataGenerator/InputText/...
### These files should represent a valid letter sequence, not random (If true random possible, then that is fine)
### A good way to gather these would be to use non-copyrighted books

## Pictures
### Download mnist data set from https://www.nist.gov/srd/nist-special-database-19, specfically the 'by_class' one
### Extract it to Datagenerator/InputLetters/by_class
### Note: The indexing of the letters is in hex

class FileLoader:
    import os
    TextPath = "";
    LetterPath = "";
    TextFileQueue = [];
    TextFileStream = [];
    LetterPaths = {};

    def __init__(self, textPath, letterPath):
        self.TextPath = textPath
        self.LetterPath = letterPath

    def CheckAndCreatePaths(self):
        print("Checking and creating file paths")
        if not self.os.path.isdir(self.TextPath):
            self.os.makedirs(self.TextPath)
        if not self.os.path.isdir(self.LetterPath):
            self.os.makedirs(self.LetterPath)
        print("Done!")

    def LoadLetterPaths(self):
        print("Loading letter paths")
        self.TextFileQueue = self.os.listdir(self.TextPath)
        if not self.TextFileQueue:
            raise Exception("Text files not found!")
        self.TextFileStream = open(self.TextPath + self.TextFileQueue[0], 'r', encoding='utf-8')
        self.TextFileQueue.pop(0)
        print("Done!")

    def GatherLetterPaths(self):
        print("Gathering letter paths")
        self.LetterPaths = {}
        for i in [*range(65, 91), *range(97, 122)]:
            self.LetterPaths[chr(i)] = {}
            hexLetter = hex(i).split('x')[-1]
            self.LetterPaths[chr(i)]['hex'] = hexLetter
        print("Done")

        print("Bind letter paths to a letters")
        # for each letter (lowercase/uppercase)
        for n in self.LetterPaths:
            self.LetterPaths[n]['paths'] = []
            # for each hsf num
            for i in range(0, 8):
                hsfPath = self.LetterPath + self.LetterPaths[n]['hex'] + "/hsf_" + str(i) + "/"
                if self.os.path.isdir(hsfPath):
                    pictures = self.os.listdir(hsfPath)
                    # for each picture in directory
                    for t in pictures:
                        self.LetterPaths[n]['paths'].append(hsfPath + t)
        print("Done")

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
        print("Create CSV...")
        self.CSVFile = open(self.FileName, 'wt', newline='')
        self.CSVWriter = self.csv.writer(self.CSVFile, delimiter=',')
        self.CSVWriter.writerow(fields)
        print("Done")

    def GenerateCSVData(self, textFileStream, letterPaths, textPath, textFileQueue):
        print("Filling CSV with data...")
        while 1:
            output = textFileStream.read(1)
            if output:
                if output in letterPaths:
                    temp = letterPaths[output]['paths']
                    # if no more pictures for that letter
                    if len(letterPaths[output]['paths']) == 0:
                        break
                    self.CSVWriter.writerow([output, letterPaths[output]['paths'][0]])
                    temp2 = letterPaths[output]['paths']
                    letterPaths[output]['paths'].pop(0)
            # in case that the stream reaches end of file, goto next file
            else:
                if len(textFileQueue) > 0:
                    textFileStream = open(textPath + textFileQueue[0], 'r', encoding='utf-8')
                    textFileQueue.pop(0)
                else:
                    break
        print("Done")

    def Finish(self):
        self.CSVFile.close()

fl = FileLoader("./DataGenerator/InputText/", "./DataGenerator/InputLetters/by_class/")
fl.CheckAndCreatePaths()
fl.LoadLetterPaths()
fl.GatherLetterPaths()

cg = CSVGenerator("data.csv", ('Letter', 'Path'))
cg.GenerateCSVData(fl.TextFileStream, fl.LetterPaths, fl.TextPath, fl.TextFileQueue)

fl.Finish()
cg.Finish()

print("Dataset generated!")
input("Press Enter to exit...")

from ProjectTools import AutoPackageInstaller as ap
from DataGenerator import TextSequence as ts

ap.CheckAndInstall("tqdm")
ap.CheckAndInstall("zipfile")

class DataExtractor:
    LETTER_RANGE = [*range(ord('a'), ord('z') + 1), *range(ord('A'), ord('Z') + 1)]
    NUMBER_RANGE = [*range(ord('0'), ord('9') + 1)]

    OutputPath = ""
    DataPath = ""
    Letters = {}
    TS = None
    DistributionPath = ""
    PrintDistribution = False
    LetterOutputIndex = {}
    IncludeNumbers = False
    IncludeLetters = False


    def __init__(self, outputPath, dataPath, textPath, distributionPath, printDistribution, includeNumbers, includeLetters):
        self.OutputPath = outputPath
        self.DataPath = dataPath
        self.TS = ts.TextSequence(textPath)
        self.DistributionPath = distributionPath
        self.PrintDistribution = printDistribution
        self.IncludeNumbers = includeNumbers
        self.IncludeLetters = includeLetters

    def ExtractSequence(self, outputFormat):
        import zipfile
        import os
        from tqdm import tqdm
        if os.path.isdir(self.OutputPath):
            return
        print("Beginning extraction (This can take a bit to start)")
        os.makedirs(self.OutputPath)
        with zipfile.ZipFile(self.DataPath, "r") as zf:
            zipInfos = self.FilterNameList(zf.infolist())
            self.CountNames(zipInfos)
            self.HandleDistributionFile(outputFormat)
            print("Extracting")
            for letter in self.Letters:
                for i in range(0, self.Letters[letter]['DistributionCount']):
                    fileInfo = zipInfos[self.Letters[letter]['StartIndex'] + i]
                    self.ExtractFile(outputFormat, zf, fileInfo, letter)
            
            

    def ExtractSpecifiedDistribution(self, outputFormat, minCount, maxCount):
        import zipfile
        import os
        from tqdm import tqdm
        if os.path.isdir(self.OutputPath):
            return
        print("Beginning extraction (This can take a bit to start)")
        os.makedirs(self.OutputPath)
        with zipfile.ZipFile(self.DataPath, "r") as zf:
            zipInfos = self.FilterNameList(zf.infolist())
            self.CountNames(zipInfos)
            self.RemoveLettersBelowLimit(minCount)
            self.HandleDistributionFile(outputFormat)
            print("Extracting")
            for letter in tqdm(iterable=self.Letters, total=len(self.Letters)):
                letterMax = min(maxCount, self.Letters[letter]['Count'])
                for i in range(0, letterMax):
                    fileInfo = zipInfos[self.Letters[letter]['StartIndex'] + i]
                    self.ExtractFile(outputFormat, zf, fileInfo, letter)
                    

    
    def ExtractFile(self, outputFormat, zipFile, fileInfo, letter):
        fileInfo.filename = str(self.Letters[letter]['Index']) + '.png'
        outputPath = self.CreateOutputPath(
            self.OutputPath, letter, outputFormat)
        zipFile.extract(member=fileInfo, path=outputPath)
        self.Letters[letter]['Index'] += 1

    def FilterNameList(self, zipInfos):
        from tqdm import tqdm

        print("Filtering zip")

        tempInfoList = []
        for i in tqdm(iterable=zipInfos, total=len(zipInfos)):
            if 'train' not in i.filename:
                if '.png' == i.filename[-4:]:
                    tempInfoList.append(i)

        return tempInfoList

    def RemoveLettersBelowLimit(self, minCount):
        tempLetters = {}
        for letter in self.Letters:
            if self.Letters[letter]['Count'] > minCount:
                tempLetters[letter] = self.Letters[letter]


        self.Letters = tempLetters

    def CountNames(self, infoList):
        from tqdm import tqdm
        print("Counting letters")

        # Populate lettercount with zeros
        allowedRange = []
        if (self.IncludeNumbers):
            allowedRange += self.NUMBER_RANGE 
        if (self.IncludeLetters):
            allowedRange += self.LETTER_RANGE
        for i in allowedRange:
            hexLetter = hex(i).split('x')[-1]
            self.Letters[chr(i)] = {}
            self.Letters[chr(i)]['HexLetter'] = hexLetter
            self.Letters[chr(i)]['StartIndex'] = -1
            self.Letters[chr(i)]['DistributionCount'] = 0
            self.Letters[chr(i)]['Index'] = 0
            self.Letters[chr(i)]['Count'] = 0

        # Count how often this matches
        for letter in tqdm(iterable=self.Letters, total=len(self.Letters)):
            for info in infoList:
                # if the letter is equal to the folder of letters of the same name
                # name[9:11] takes the substring from character 9 to 11
                if self.Letters[letter]['HexLetter'] == info.filename[9:11]:
                    if self.Letters[letter]['StartIndex'] == -1:
                        self.Letters[letter]['StartIndex'] = infoList.index(
                            info)
                    self.Letters[letter]['Count'] += 1

    def HandleDistributionFile(self, format):
        if self.PrintDistribution:
            self.CountDistribution()
            self.PrintDistributionCSV(format)

    def CountDistribution(self):
        while 1:
            extractionLetter = self.TS.GetNextLetter()
            if extractionLetter in self.Letters:
                if self.Letters[extractionLetter]['DistributionCount'] > self.Letters[extractionLetter]['Count']:
                    return
                else:
                    self.Letters[extractionLetter]['DistributionCount'] += 1

    def PrintDistributionCSV(self, format):
        import csv
        with open(self.DistributionPath, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['letter', 'index', 'count'])
            for letter in self.Letters:
                writer.writerow([letter, self.GetIndexFormat(letter, format), self.Letters[letter]['DistributionCount']])
                    

    def CreateOutputPath(self, basePath, letter, format):
        return basePath + self.GetIndexFormat(letter, format)

    def GetIndexFormat(self, letter, format):
        index = ''
        if format == 'Letter':
            if str.islower(letter):
                index = '_'
            index += letter
        elif format == 'Number':
            index += str(ord(letter))
        elif format == 'ZeroIndexed':
            if not letter in self.LetterOutputIndex:
                self.LetterOutputIndex[letter] = {}
                self.LetterOutputIndex[letter]['index'] = len(self.LetterOutputIndex) - 1
            index += str(self.LetterOutputIndex[letter]['index'])
        else:
            raise Exception("Invalid output format: " + format + " Should be 'Letter' or 'Number'")
        return index

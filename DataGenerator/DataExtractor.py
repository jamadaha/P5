import AutoPackageInstaller as ap

ap.CheckAndInstall("tqdm")
ap.CheckAndInstall("zipfile")


class DataExtractor:
    OutputPath = ""
    DataPath = ""
    Letters = {}
    TS = None

    def __init__(self, outputPath, dataPath, textSequence):
        self.OutputPath = outputPath
        self.DataPath = dataPath
        self.TS = textSequence

    def Extract(self, outputFormat):
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
            print("Extracting")
            while 1:
                extractionLetter = self.TS.GetNextLetter()
                if extractionLetter in self.Letters:

                    if self.Letters[extractionLetter]['Index'] > self.Letters[extractionLetter]['Count']:
                        break

                    fileInfo = zipInfos[self.Letters[extractionLetter]
                                        ['StartIndex'] + self.Letters[extractionLetter]['Index']]
                    fileInfo.filename = str(
                        self.Letters[extractionLetter]['Index']) + '.png'

                    outputPath = self.CreateOutputPath(
                        self.OutputPath, extractionLetter, outputFormat)

                    zf.extract(member=fileInfo, path=outputPath)
                    self.Letters[extractionLetter]['Index'] += 1

    def FilterNameList(self, zipInfos):
        from tqdm import tqdm

        print("Filtering zip")

        tempInfoList = []
        for i in tqdm(iterable=zipInfos, total=len(zipInfos)):
            if 'train' not in i.filename:
                if '.png' == i.filename[-4:]:
                    tempInfoList.append(i)

        return tempInfoList

    def CountNames(self, infoList):
        from tqdm import tqdm
        import os
        print("Counting letters")

        # Populate lettercount with zeros
        for i in [*range(65, 91), *range(97, 122)]:
            hexLetter = hex(i).split('x')[-1]
            self.Letters[chr(i)] = {}
            self.Letters[chr(i)]['HexLetter'] = hexLetter
            self.Letters[chr(i)]['StartIndex'] = -1
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

    def CreateOutputPath(self, basePath, letter, format):
        outputPath = basePath
        if format == 'Letter':
            if str.islower(letter):
                outputPath = basePath + '_'
            outputPath += letter
        else:
            if format == 'Number':
                outputPath += str(ord(letter))
            else:
                raise Exception("Invalid output format: " + format + " Should be 'Letter' or 'Number'")
        return outputPath

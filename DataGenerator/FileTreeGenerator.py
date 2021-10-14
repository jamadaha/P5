from ProjectTools import AutoModule as am

am.CheckAndInstall("tqdm")
am.CheckAndInstall("pandas")
am.CheckAndInstall("shutil")

class FileTreeGenerator:
    import pandas
    import os
    import shutil
    import tqdm
    CSVPath = ""
    CSVFile = None
    OutputPath = ""

    def __init__(self, csvPath, outputPath):
        self.CSVPath = csvPath
        self.CSVFile = open(csvPath, 'r')
        self.OutputPath = outputPath

    def Generate(self):
        print("Checking data folder ... ", end="")
        data = self.pandas.read_csv(self.CSVPath)
        counts = {}
        if not self.os.path.isdir(self.OutputPath):
            print("Does not exist, beginning data tree generation ... ")
            self.os.makedirs(self.OutputPath)
        else:
            return

        for row in self.tqdm([*data.iterrows()]):
            letter = row[1][0]
            # Renames the lowercase letters to have a (_) prefix, to differentiate them on Windows
            if str.islower(letter):
                letter = '_' + letter


            if not letter in counts:
                counts[letter] = 0
                self.os.makedirs(self.OutputPath + letter + '/')
            else:
                counts[letter] += 1

            self.shutil.copyfile(row[1][1], self.OutputPath + letter + '/' + str(counts[letter]) + '.png')

        print("Done")
        self.Finish()

    def Finish(self):
        self.CSVFile.close()

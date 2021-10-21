from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tqdm")
ap.CheckAndInstall("pandas")
ap.CheckAndInstall("shutil")

class FileTreeGenerator:
    import pandas
    import os
    import shutil
    from tqdm import tqdm
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
        letterCounter = {}
        counter = 0
        if not self.os.path.isdir(self.OutputPath):
            print("Does not exist, beginning data tree generation ... ")
            self.os.makedirs(self.OutputPath)
        else:
            return

        for row in self.tqdm([*data.iterrows()]):
            letter = ord(row[1][0])
            # Renames the lowercase letters to have a (_) prefix, to differentiate them on Windows
            #if str.islower(letter):
            #    letter = '_' + letter


            if not str(letter) in counts:
                letterCounter[letter] = counter
                counts[str(letter)] = 0
                self.os.makedirs(self.OutputPath + str(letterCounter[letter]) + '/')
                counter += 1
            else:
                counts[str(letter)] += 1

            self.shutil.copyfile(row[1][1], self.OutputPath + str(letterCounter[letter]) + '/' + str(counts[str(letter)]) + '.png')

        print("Done")
        self.Finish()

    def Finish(self):
        self.CSVFile.close()

from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("csv")
ap.CheckAndInstall("time")

class Logger:
    OutputDir = ''
    Name = ''

    def __init__(self, outputDir: str, name: str) -> None:
        self.OutputDir = outputDir
        self.Name = name

    def InitCSV(self, data: list) -> None:
        import os
        if not os.path.isdir(self.OutputDir):
            os.mkdir(self.OutputDir)
        self.__WriteToCSV('w', data)

    def AppendToCSV(self, data: list) -> None:
        self.__WriteToCSV('a+', data)
    
    def __WriteToCSV(self, writeMode: str, data: list) -> None:
        import csv
        import time
        with open(self.OutputDir + self.Name + '.csv', writeMode) as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(data)
                
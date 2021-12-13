from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("csv")
ap.CheckAndInstall("time")
ap.CheckAndInstall("os")

import os
import time
import csv

class CSVLogger:
    OutputDir = ''
    Name = ''

    def __init__(self, outputDir: str, name: str) -> None:
        self.OutputDir = outputDir
        self.Name = name

    def InitCSV(self, data: list) -> None:
        if not os.path.isdir(self.OutputDir):
            os.makedirs(self.OutputDir, exist_ok=True)
        tmpData = data.copy()
        tmpData.insert(0, 'Time')
        self.__WriteToCSV('w', tmpData)

    def AppendToCSV(self, data: list) -> None:
        tmpData = data.copy()
        tmpData.insert(0, time.strftime("%H:%M:%S", time.gmtime(time.time())))
        self.__WriteToCSV('a+', tmpData)
    
    def __WriteToCSV(self, writeMode: str, data: list) -> None:
        with open(self.OutputDir + self.Name + '.csv', writeMode) as file:
                csv_writer = csv.writer(file, delimiter=',', lineterminator='\n')
                csv_writer.writerow(data)

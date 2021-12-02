from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("csv")
ap.CheckAndInstall("time")

class CSVLogger:
    OutputDir = ''
    Name = ''

    def __init__(self, outputDir: str, name: str) -> None:
        self.OutputDir = outputDir
        self.Name = name

    def InitCSV(self, data: list) -> None:
        import os
        if not os.path.isdir(self.OutputDir):
            os.makedirs(self.OutputDir, exist_ok=True)
        tmpData = data.copy()
        tmpData.insert(0, 'Time')
        self.__WriteToCSV('w', tmpData)

    def AppendToCSV(self, data: list) -> None:
        import time
        tmpData = data.copy()
        tmpData.insert(0, time.strftime("%H:%M:%S", time.gmtime(time.time())))
        self.__WriteToCSV('a+', tmpData)
    
    def __WriteToCSV(self, writeMode: str, data: list) -> None:
        import csv
        with open(self.OutputDir + self.Name + '.csv', writeMode) as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(data)

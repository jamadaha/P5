class Logger:
    def __init__(self, outputDir, name):
        self.OutputDir = outputDir
        self.Name = name

    def InitCSV(self, data) -> None:
        import os
        if not os.path.isdir(self.OutputDir):
            os.mkdir(self.OutputDir)
        self.__WriteToCSV('w', data)

    def AppendToCSV(self, data) -> None:
        self.__WriteToCSV('a+', data)
    
    def __WriteToCSV(self, writeMode, data) -> None:
        with open(self.OutputDir, writeMode) as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(data)
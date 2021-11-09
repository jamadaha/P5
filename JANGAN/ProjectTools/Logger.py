class Logger:
    def __init__(self):
        pass

    def InitCSV(self, data) -> None:
        if self.WriteToCSV:
            with open(self.CSVPath, 'w') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(data)

    def AppendToCSV(self, data) -> None:
        if self.WriteToCSV:
            with open(self.CSVPath, 'a+') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(data)
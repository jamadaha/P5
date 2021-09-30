
class CSVGenerator:
    import csv
    FileName = ""
    Fields = ()
    CSVFile = {}
    CSVWriter = {}

    def __init__(self, fileName, fields):
        self.FileName = fileName
        self.Fields = fields
        self.CSVFile = open(self.FileName, 'wt', newline='')
        self.CSVWriter = self.csv.writer(self.CSVFile, delimiter=',')
        self.CSVWriter.writerow(fields)

    def GenerateCSVData(self, textFileStream, letterPaths, textPath, textFileQueue):
        print("Filling CSV with data ... ", end="")
        while 1:
            output = textFileStream.read(1)
            if output:
                if output in letterPaths:
                    # if no more pictures for that letter
                    if len(letterPaths[output]['paths']) == 0:
                        break
                    self.CSVWriter.writerow(
                        [output, letterPaths[output]['paths'][0]])
                    letterPaths[output]['paths'].pop(0)
            # in case that the stream reaches end of file, goto next file
            else:
                if len(textFileQueue) > 0:
                    textFileStream = open(
                        textPath + textFileQueue[0], 'r', encoding='utf-8')
                    textFileQueue.pop(0)
                else:
                    break
        self.Finish()
        print("Done")

    def Finish(self):
        self.CSVFile.close()
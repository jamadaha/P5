from ProjectTools import ConfigHelper as cfg

from CSVGenerator import CSVGenerator
from FileLoader import FileLoader
from FileTreeGenerator import FileTreeGenerator

fl = FileLoader(
    cfg.GetStringValue("DATAGENERATOR","TextPath"),
    cfg.GetStringValue("DATAGENERATOR","LetterPath"),
    cfg.GetStringValue("DATAGENERATOR","LetterDownloadURL"),
    cfg.GetJsonValue("DATAGENERATOR","TextDownloadURLS"),
    cfg.GetStringValue("DATAGENERATOR","TempDownloadLetterPath"))
fl.ImportAllData()

cg = CSVGenerator(
    cfg.GetStringValue("DATAGENERATOR","CSVFileName"),
    ('Letter', 'Path'))
cg.GenerateCSVData(fl.TextFileStream, fl.LetterPaths,
                   fl.TextPath, fl.TextFileQueue)

fl.Finish()

ftg = FileTreeGenerator(
    cfg.GetStringValue("DATAGENERATOR","CSVFileName"),
    cfg.GetStringValue("DATAGENERATOR","OutputLettersPath"))
ftg.Generate()

print("Dataset generated!")
input("Press Enter to exit...")

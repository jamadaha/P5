import sys
sys.path.append('./ProjectTools')

import ConfigHelper as cfg


from FileImporter import FileImporter
from DataExtractor import DataExtractor
from TextSequence import TextSequence


fl = FileImporter(
    cfg.GetStringValue("DATAGENERATOR","TextPath"),
    cfg.GetStringValue("DATAGENERATOR","LetterDownloadURL"),
    cfg.GetJsonValue("DATAGENERATOR","TextDownloadURLS"),
    cfg.GetStringValue("DATAGENERATOR","TempDownloadLetterPath"),
    cfg.GetStringValue("DATAGENERATOR", "TempDownloadLetterFileName"))
fl.ImportFiles()

ts = TextSequence(
    cfg.GetStringValue("DATAGENERATOR", "TextPath"))

ftg = DataExtractor(
    cfg.GetStringValue("DATAGENERATOR", "OutputLettersPath"),
    cfg.GetStringValue("DATAGENERATOR", "TempDownloadLetterPath") +
    cfg.GetStringValue("DATAGENERATOR", "TempDownloadLetterFileName"),
    ts)
ftg.Extract()

print("Dataset generated!")
input("Press Enter to exit...")

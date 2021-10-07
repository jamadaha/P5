import configparser
import json
config = configparser.ConfigParser()
config.read('config.ini')

from CSVGenerator import CSVGenerator
from FileLoader import FileLoader
from FileTreeGenerator import FileTreeGenerator

fl = FileLoader(
    config["DATAGENERATOR"]["TextPath"].strip('"'), 
    config["DATAGENERATOR"]["LetterPath"].strip('"'), 
    config["DATAGENERATOR"]["LetterDownloadURL"].strip('"'), 
    json.loads(config["DATAGENERATOR"]["TextDownloadURLS"]),
    config["DATAGENERATOR"]["TempDownloadLetterPath"].strip('"'))
fl.CheckAndCreatePaths()
fl.LoadLetterPaths()
fl.GatherLetterPaths()

cg = CSVGenerator(
    config["DATAGENERATOR"]["CSVFileName"].strip('"'), 
    ('Letter', 'Path'))
cg.GenerateCSVData(fl.TextFileStream, fl.LetterPaths,
                   fl.TextPath, fl.TextFileQueue)

fl.Finish()

ftg = FileTreeGenerator(
    config["DATAGENERATOR"]["CSVFileName"].strip('"'), 
    config["DATAGENERATOR"]["OutputLettersPath"].strip('"'))
ftg.Generate()

print("Dataset generated!")
input("Press Enter to exit...")
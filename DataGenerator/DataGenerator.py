import ConfigHelper as cfg

import json
from CSVGenerator import CSVGenerator
from FileLoader import FileLoader
from FileTreeGenerator import FileTreeGenerator

fl = FileLoader(
    cfg.config["DATAGENERATOR"]["TextPath"].strip('"'), 
    cfg.config["DATAGENERATOR"]["LetterPath"].strip('"'), 
    cfg.config["DATAGENERATOR"]["LetterDownloadURL"].strip('"'), 
    json.loads(cfg.config["DATAGENERATOR"]["TextDownloadURLS"]),
    cfg.config["DATAGENERATOR"]["TempDownloadLetterPath"].strip('"'))
fl.CheckAndCreatePaths()
fl.LoadLetterPaths()
fl.GatherLetterPaths()

cg = CSVGenerator(
    cfg.config["DATAGENERATOR"]["CSVFileName"].strip('"'), 
    ('Letter', 'Path'))
cg.GenerateCSVData(fl.TextFileStream, fl.LetterPaths,
                   fl.TextPath, fl.TextFileQueue)

fl.Finish()

ftg = FileTreeGenerator(
    cfg.config["DATAGENERATOR"]["CSVFileName"].strip('"'), 
    cfg.config["DATAGENERATOR"]["OutputLettersPath"].strip('"'))
ftg.Generate()

print("Dataset generated!")
input("Press Enter to exit...")
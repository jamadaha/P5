from CSVGenerator import CSVGenerator
from FileLoader import FileLoader
from FileTreeGenerator import FileTreeGenerator

fl = FileLoader("./DataGenerator/InputText/", "./DataGenerator/InputLetters/by_class/", "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
                [['AChristmasCarol', 'https://www.gutenberg.org/files/46/46-0.txt'], ['PrideandPrejudice', 'https://www.gutenberg.org/files/1342/1342-0.txt']])
fl.CheckAndCreatePaths()
fl.LoadLetterPaths()
fl.GatherLetterPaths()

cg = CSVGenerator("./DataGenerator/data.csv", ('Letter', 'Path'))
cg.GenerateCSVData(fl.TextFileStream, fl.LetterPaths,
                   fl.TextPath, fl.TextFileQueue)

fl.Finish()

ftg = FileTreeGenerator("./DataGenerator/data.csv", "./DataGenerator/Data/")
ftg.Generate()

print("Dataset generated!")
input("Press Enter to exit...")
from CSVGenerator import CSVGenerator
from FileLoader import FileLoader
from FileTreeGenerator import FileTreeGenerator

fl = FileLoader("./DataGenerator/InputText/", "./DataGenerator/InputLetters/by_class/", "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
                [['AChristmasCarol', 'https://www.gutenberg.org/files/46/46-0.txt'], ['PrideandPrejudice', 'https://www.gutenberg.org/files/1342/1342-0.txt']])
fl.CheckAndCreatePaths()
fl.LoadLetterPaths()
fl.GatherLetterPaths()

cg = CSVGenerator("./DataGenerator/data.csv", ('Letter', 'Path'))
cg.GenerateCSVData(fl.TextFileStream, fl.LetterPaths,
                   fl.TextPath, fl.TextFileQueue)

fl.Finish()

ftg = FileTreeGenerator("./DataGenerator/data.csv", "./DataGenerator/Data/")
ftg.Generate()

print("Dataset generated!")
input("Press Enter to exit...")
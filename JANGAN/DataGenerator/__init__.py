from . import FileImporter as fi
from . import DataExtractor as de
from . import TextSequence as ts

class DataGenerator():
    FileImporter_TextPath = ""
    FileImporter_LetterDownloadURL = ""
    FileImporter_TextDownloadURLS = {}
    FileImporter_TempDownloadLetterPath = ""
    FileImporter_TempDownloadLetterFileName = ""

    TextSequence_TextPath = ""

    DataExtractor_OutputLetterPath = ""
    DataExtractor_TempDownloadLetterPath = ""
    DataExtractor_TempDownloadLetterFileName = ""
    DataExtractor_MinimumLetterCount = 0
    DataExtractor_MaximumLetterCount = 0
    DataExtractor_OutputLetterFormat = ""

    __FileImporter = None
    __DataExtractor = None
    __TextSequence = None
        
    def ConfigureFileImporter(self, fi_TextPath, fi_LetterDownloadURL, fi_TextDownloadURLS, fi_TempDownloadLetterPath, fi_TempDownloadLetterFileName):
        print("Configuring File Importer")

        self.FileImporter_TextPath = fi_TextPath
        self.FileImporter_LetterDownloadURL = fi_LetterDownloadURL
        self.FileImporter_TextDownloadURLS = fi_TextDownloadURLS
        self.FileImporter_TempDownloadLetterPath = fi_TempDownloadLetterPath
        self.FileImporter_TempDownloadLetterFileName = fi_TempDownloadLetterFileName

    def ConfigureTextSequence(self, ts_textPath):
        print("Configuring Test Sequence")

        self.TextSequence_TextPath = ts_textPath

    def ConfigureDataExtractor(self, de_OutputLettersPath, de_TempDownloadLetterPath, de_TempDownloadLetterFileName, de_MinimumLetterCount, de_MaximumLetterCount, de_OutputLetterFormat):
        print("Configuring Data Extractor")

        self.DataExtractor_OutputLetterPath = de_OutputLettersPath
        self.DataExtractor_TempDownloadLetterPath = de_TempDownloadLetterPath
        self.DataExtractor_TempDownloadLetterFileName = de_TempDownloadLetterFileName
        self.DataExtractor_MinimumLetterCount = de_MinimumLetterCount
        self.DataExtractor_MaximumLetterCount = de_MaximumLetterCount
        self.DataExtractor_OutputLetterFormat = de_OutputLetterFormat

    def GenerateData(self):
        print("Generating dataset...")

        self.__FileImporter = fi.FileImporter(
            self.FileImporter_TextPath,
            self.FileImporter_LetterDownloadURL,
            self.FileImporter_TextDownloadURLS,
            self.FileImporter_TempDownloadLetterPath,
            self.FileImporter_TempDownloadLetterFileName)

        self.__FileImporter.ImportFiles()

        self.__TextSequence = ts.TextSequence(
            self.TextSequence_TextPath)

        self.__DataExtractor = de.DataExtractor(
            self.DataExtractor_OutputLetterPath,
            self.DataExtractor_TempDownloadLetterPath +
            self.DataExtractor_TempDownloadLetterFileName,
            self.__TextSequence)

        if self.DataExtractor_MinimumLetterCount == 0 and self.DataExtractor_MaximumLetterCount == 0:
            self.__DataExtractor.ExtractSequence(
                self.DataExtractor_OutputLetterFormat
            )
        else:
            self.__DataExtractor.ExtractSpecifiedDistribution(
                self.DataExtractor_OutputLetterFormat,
                self.DataExtractor_MinimumLetterCount,
                self.DataExtractor_MaximumLetterCount
            )

        print("Dataset generated!")

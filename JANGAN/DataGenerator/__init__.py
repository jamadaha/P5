from DataGenerator import FileImporter as fi
from DataGenerator import DataExtractor as de

class DataGenerator():
    __LetterOutputFormat = ""
    __LetterOutputMinCount = 0
    __LetterOutputMaxCount = 0

    __FileImporter = None
    __DataExtractor = None

    def __init__(
        self, 
        letterDownloadURL, 
        letterDownloadPath,
        letterDownloadName,
        letterOutputPath,
        letterOutputFormat,
        letterOutputMinCount,
        letterOutputMaxCount,
        textDownloadURLS,
        textPath,
        includeNumbers,
        includeLetters) -> None:

        self.__LetterOutputFormat = letterOutputFormat
        self.__LetterOutputMinCount = letterOutputMinCount
        self.__LetterOutputMaxCount = letterOutputMaxCount

        self.__FileImporter = fi.FileImporter(
            textPath,
            letterDownloadURL,
            textDownloadURLS,
            letterDownloadPath,
            letterDownloadName)

        self.__FileImporter.ImportFiles()

        self.__DataExtractor = de.DataExtractor(
            letterOutputPath,
            letterDownloadPath +
            letterDownloadName,
            textPath,
            includeNumbers,
            includeLetters)

    def GenerateData(self):
        print("Generating dataset...")

        if self.__LetterOutputMinCount == 0 and self.__LetterOutputMaxCount == 0:
            self.__DataExtractor.ExtractSequence(
                self.__LetterOutputFormat
            )
        else:
            self.__DataExtractor.ExtractSpecifiedDistribution(
                self.__LetterOutputFormat,
                self.__LetterOutputMinCount,
                self.__LetterOutputMaxCount
            )

        print("Dataset generated!")

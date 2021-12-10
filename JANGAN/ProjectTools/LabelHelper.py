class LabelHelper:
    LETTER_RANGE = [*range(ord('a'), ord('z') + 1), *range(ord('A'), ord('Z') + 1)]
    NUMBER_RANGE = [*range(ord('0'), ord('9') + 1)]

    def LetterLabels(self):
        letters = {}
        allowedRange = self.LETTER_RANGE
        for i in allowedRange:
            hexLetter = hex(i).split('x')[-1]
            letters[chr(i)] = {}
            letters[chr(i)]['HexLetter'] = hexLetter
            letters[chr(i)]['StartIndex'] = -1
            letters[chr(i)]['DistributionCount'] = 0
            letters[chr(i)]['Index'] = 0
            letters[chr(i)]['Count'] = 0

        return letters

    def NumberLabels(self):
        letters = {}
        allowedRange = self.NUMBER_RANGE
        for i in allowedRange:
            hexLetter = hex(i).split('x')[-1]
            letters[chr(i)] = {}
            letters[chr(i)]['HexLetter'] = hexLetter
            letters[chr(i)]['StartIndex'] = -1
            letters[chr(i)]['DistributionCount'] = 0
            letters[chr(i)]['Index'] = 0
            letters[chr(i)]['Count'] = 0

        return letters

    def LetterLabelArray(self):
        letters = []
        allowedRange = self.LETTER_RANGE
        for l in allowedRange:
            letters.append(chr(l))
            
        return letters

    def NumberLabelArray(self):
        letters = []
        allowedRange = self.NUMBER_RANGE
        for l in allowedRange:
            letters.append(chr(l))

        return letters


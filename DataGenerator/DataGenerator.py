# Get 2 flows of data - Text files and pictures

## Text files
### Move .txt files into the folder DataGenerator/InputText/...
### These files should represent a valid letter sequence, not random (If true random possible, then that is fine)
### A good way to gather these would be to use non-copyrighted books
import os

textPath = "InputText/"

print("Checking whether text file directory exists...")

if not os.path.isdir(textPath):
    raise Exception("Missing directory " + textPath)
    
print("It does")

textFileQueue = os.listdir(textPath)
textFileStream = open(textPath + textFileQueue[0], 'r', encoding='utf-8')
textFileQueue.pop(0)

## Pictures
### Download mnist data set from https://www.nist.gov/srd/nist-special-database-19, specfically the 'by_class' one
### Extract it to Datagenerator/InputLetters/by_class
### Note: The indexing of the letters is in hex
letterSuperPath = "InputLetters/by_class/"

print("Checking whether text file directory exists...")

# check that directory exists
if not os.path.isdir(letterSuperPath):
    raise Exception("Missing directory " + letterSuperPath)

print("It does")

print("Gathering letter paths...")

letterPath = {}
for i in [*range(65, 91), *range(97, 122)]:
    letterPath[chr(i)] = {}
    hexLetter = hex(i).split('x')[-1]
    letterPath[chr(i)]['hex'] = hexLetter

print("Done")

### Add paths to individual pictures

print("Gathering picture paths...")

# for each letter (lowercase/uppercase)
for n in letterPath:
    letterPath[n]['paths'] = []
    # for each hsf num
    for i in range(0, 8):
        hsfPath = letterSuperPath + letterPath[n]['hex'] + "/hsf_" + str(i) + "/"
        if os.path.isdir(hsfPath):
            pictures = os.listdir(hsfPath)
            # for each picture in directory
            for t in pictures:
                letterPath[n]['paths'].append(hsfPath + t)

print("Done")

# Generate CSV file
## Create file
import csv

print("Create CSV...")

csvFile = open('data.csv', 'wt', newline='')
fields = ('Letter', 'Path')
writer = csv.writer(csvFile, delimiter=',')

writer.writerow(fields)

print("Done")


## Fill file with data
print("Filling CSV with data...")

while 1:
    output = textFileStream.read(1)
    if output:
        if output in letterPath:
            temp = letterPath[output]['paths']
            # if no more pictures for that letter
            if len(letterPath[output]['paths']) == 0:
                break
            writer.writerow([output, letterPath[output]['paths'][0]])
            temp2 = letterPath[output]['paths']
            letterPath[output]['paths'].pop(0)
    # in case that the stream reaches end of file, goto next file
    else:
        if len(textFileQueue) > 0:
            textFileStream = open(textPath + textFileQueue[0], 'r', encoding='utf-8')
            textFileQueue.pop(0)
        else:
            break

csvFile.close()

print("Done")

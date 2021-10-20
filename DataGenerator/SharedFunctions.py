import AutoPackageInstaller as ap

ap.CheckAndInstall("tqdm")
ap.CheckAndInstall("requests")

# Returns path to file
def DownloadIfNotExist(url, path, fileName) -> str:
    import os
    if not os.path.isfile(path + fileName):
        if not os.path.isdir(path):
            os.makedirs(path)
        return Download(url, path, fileName)
    else:
        return path + fileName

# Returns path to file
def Download(url, path, fileName) -> str:
    import requests
    import os
    from tqdm import tqdm

    print("Downloading " + fileName)

    response = requests.get(url, stream=True)

    fileLoc = path + fileName
    fileSize = GetFileSize(response)
    chunkSize = 1024
    progressBar = tqdm(total=fileSize, unit='iB', unit_scale=True)
    with open(fileLoc, 'wb') as writeStream:
        for chunk in response.iter_content(chunkSize):
            progressBar.update(chunkSize)
            writeStream.write(chunk)
    progressBar.close()

    return path + fileName


def GetFileSize(response):
    return int(response.headers.get('content-length', 0))

def PrintIndented(indentAmount, text):
    for i in range(0, indentAmount + 1):
        print("\t", end="")
    print(text)

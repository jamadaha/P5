# Returns path to file
def Download(url, path, fileName) -> str:
    import requests
    import os
    from tqdm import tqdm

    print("Downloading " + fileName + " ... ")

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

    print("Done")
    return path + fileName


def GetFileSize(response):
    return int(response.headers.get('content-length', 0))

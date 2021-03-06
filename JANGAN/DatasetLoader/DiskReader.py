from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tqdm")
ap.CheckAndInstall("os")
ap.CheckAndInstall("cv2","opencv-python") #cv2
ap.CheckAndInstall("numpy")

import os
import cv2
import numpy as np

class DiskReader():
    Dir = ""
    LableID = -1
    Images = []
    Labels = []
    DataSize = 0
    ImageSize = ()
    FormatImages = True

    def __init__(self, dir, labelID, imageSize, formatImages = True):
        self.Dir = dir
        self.LableID = labelID
        self.ImageSize = imageSize
        self.FormatImages = formatImages

    def ReadImagesAndLabelsFromDisc(self):
        features, labels = [], []

        if os.path.isdir(self.Dir):
            for img_name in os.listdir(self.Dir):
                img = cv2.imread(os.path.join(self.Dir, img_name), cv2.IMREAD_GRAYSCALE)
                if self.FormatImages == True:
                    img = cv2.bitwise_not(img)
                img = cv2.resize(img, self.ImageSize)
                features.append(img)
                labels.append(self.LableID)
                self.DataSize += 1
        else:
            return None

        self.Images = np.array(features, dtype=np.float32)
        self.Labels = np.array(labels, dtype=np.float32)
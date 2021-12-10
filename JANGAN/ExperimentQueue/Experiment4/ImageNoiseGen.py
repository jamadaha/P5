import numpy as np
import cv2
import random

class ImageNoiseGen:
    def __init__(self):
        self.transformId = 0
    def ApplyNoise(self, image, randomNumber):

        rand = random.randint(0, 10) #Includes endpoints

        actionsToPerform = 0
        if rand<4:
            actionsToPerform = 1
        elif rand<8:
            actionsToPerform = 2
        else:
            actionsToPerform = 3

        for i in range(actionsToPerform):
            
            #cv2.imshow("Original image: ", image)
            image = self.ApplyRandomTransformation(image)
            #cv2.imshow("Transformed image: ", image)

            #self.transformId = self.transformId + 1
            #saveImg = image.astype(np.uint8)
            #savedSuccess = cv2.imwrite(str(f"../../transformedImages/Transformed_{self.transformId}.jpg"), saveImg)
            #savedSuccess = cv2.imwrite(str(f"C:/Users/Henrik/source/repos/jamadaha/transformedImages/Transformed_{self.transformId}.jpg"), saveImg)
            
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        return image

    def ApplyRandomTransformation(self, image):
        actionId = random.randint(0, 2)
        if actionId==0:
            return self.Rotate(image, random.randint(-10, 10))
        elif actionId==1:
            return self.Zoom(image, random.randint(95, 110)/100)
        elif actionId==2:
            return self.Stretch(image,
                                random.randint(95, 110)/100,
                                random.randint(95, 110)/100
                                )

    def Rotate(self, image, degrees):
        # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, degrees, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def Zoom(self, image, factor):
        #return cv2.resize(image, None, fx=factor, fy=factor)
    
        newImage = cv2.resize(image, None, fx=factor, fy=factor)
        newImage = self.__SetResolutionWithCropOrPad(image, 28, 28, [0])
        return newImage

    def Stretch(self, image, zoomFactorX, zoomFactorY):
        newImage = cv2.resize(image, None, fx=zoomFactorX, fy=zoomFactorY)
        newImage = self.__SetResolutionWithCropOrPad(image, 28, 28, [0])
        return newImage

    def __SetResolutionWithCropOrPad(self, image, width, height, backgroundColor):


        #sizeDif = image.shape - (width, height) # Negative = image currently too big

        return self.__PadImage(image, width, height, backgroundColor)

    def __PadImage(self, image, width, height, fillColor):
        # https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape

        oldHeight, oldWidth = image.shape

        # create new image of desired size and color (blue) for padding
        result = np.full((height, width), fillColor, dtype=image.dtype)

        # compute center offset
        centerX = (width-oldWidth) // 2
        centerY = (height-oldHeight) // 2

        # copy img image into center of result image
        result[centerY:centerY+oldHeight, 
               centerX:centerX+oldWidth] = image

        return result
        




import Generator
import Discriminator

class GAN(object):
    """description of class"""

    def __init__(self, *args, **kwargs):
        self.Generator = Generator();
        self.Discriminator = Discriminator();
        return super().__init__(*args, **kwargs)

    def Train(image, label):
        print("Train")
    
    def Generate(label):
        print("Generate")

    def Discriminate(image, label):
        print("Discriminate")













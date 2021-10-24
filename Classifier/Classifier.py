import DataLoader
import tensorflow

class Classifier(object):
    """description of class"""
    
    def __init__(self, path: str):
        self.data_dir = path

    def __load_data__(self, path) :
        self.data_loader = DataLoader(path)
        try: 
                self.data = self.data_loader.load_from_dir()
        except:
                return NotADirectoryError("An Error occured when trying to load data from path. \n")
        else: 
                print("Succesfully loaded data from path. \n")

        return self.data
    
  
    def __set_up_model__(self, layers = [tensorflow.keras.layers.Flatten(input_shape=(28,28)), 
                                                  tensorflow.keras.layers.Dense(128, activation='relu'), 
                                                  tensorflow.keras.layers.Dense(10)], model_name = "simpleModel") :
        if (self.data == None) : 
            print("Data not loaded before setting up model. Trying to load data from default path. \n")
            self.__load_data__()

        self.model = tensorflow.keras.Sequential(layers, model_name)
        
    def create_and_compile_model(self, layers, model_name: str):
        self.__set_up_model__(layers, model_name)
        self.model.compile(optimize='adam', loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    def fit_model(self, train_images, train_labels, epochs: int): 
        __set_up_model(train_image, train_labels, epochs)
        self.model.fit()
        return self.model.evaluate(test_images, test_labels, verbose=2)



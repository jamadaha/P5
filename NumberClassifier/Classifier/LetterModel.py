from tensorflow.keras import Sequential
from Classifier.LayerConfigObject import LayerConfigObject
from Classifier.CompilerConfigObject import CompilerConfigObject


class LetterModel():
    """Subclass of tensorflor.keras.Model"""

    def __init__(self, layers: LayerConfigObject):
        self.sequential = Sequential(layers = layers.layers)

    def compile(self, compile_config: CompilerConfigObject):
        self.sequential.compile(
            optimizer = compile_config.optimizer, 
            loss=compile_config.loss, 
            metrics = compile_config.metrics, 
            weighted_metrics = compile_config.weighted_metrics, 
            run_eagerly = compile_config.run_eagerly,
            steps_per_execution = compile_config.steps_per_execution
            )
    

        
        
        

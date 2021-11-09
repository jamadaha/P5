import tensorflow

class CompilerConfigObject(object):
    """description of class"""
    
    def __init__(self):
        self.optimizer = "adam"
        self.loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = ['accuracy']
        self.loss_weights = None
        self.weighted_metrics = None
        self.run_eagerly = None
        self.steps_per_execution = None



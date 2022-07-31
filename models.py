'''Initialize custom model for training and pruning.'''

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds  



class maskInit(tf.keras.initializers.Initializer):

  def __init__(self, mask = None, 
               preinit_weights = None,
               initializer= tf.keras.initializers.HeNormal()):
    '''
    Returns custom weight initializer with mask.  

    Parameters:  
      - mask: ndarray of mask
      - preinit_weights: weights
      - initializer: tf.keras.initializers object
    '''
    self.mask = mask
    self.initializer = initializer
    self.preinit_weights = preinit_weights

  def __call__(self, shape, dtype = None):

    out = None

    if self.preinit_weights is not None and self.mask is not None:

        out = tf.math.multiply(self.mask, self.preinit_weights)

    elif self.preinit_weights is None and self.mask is not None: 

        out = tf.math.multiply(self.mask, self.initializer(shape)) 

    elif self.preinit_weights is None and self.mask is None:
        # first initialization of model without any weights/masks
        out = self.initializer(shape)

    assert out is not None

    return out  


class customLinear(tf.keras.layers.Layer):  

    def __init__(self, num_out,
                activation = 'relu',
                initializer = tf.keras.initializers.HeNormal(),
                BatchNorm=None, 
                Dropout=None,
                mask=None,
                preinit_weights=None):
        '''
        Returns a custom linear layer.

        Parameters:
            - num_out: number of output nodes
            - activation: activation function
            - BatchNorm: bool, whether to use batch normalization
            - Dropout: float, dropout rate
        '''
        super(customLinear, self).__init__()
        self.num_out = num_out
        self.activation = activation
        self.BatchNorm = BatchNorm
        self.Dropout = Dropout
        self.initializer = initializer
        self.mask = mask
        self.preinit_weights = preinit_weights

    def call(self, inputs):
        out = tf.keras.layers.Dense(self.num_out,
                                    activation=self.activation,
                                    kernel_initializer=maskInit(self.mask,self.preinit_weights,
                                    self.initializer))(inputs)
        if self.BatchNorm is not None:
            out = tf.keras.layers.BatchNormalization()(out)
        if self.Dropout is not None:
            out = tf.keras.layers.Dropout(self.Dropout)(out)

        return out



def makeModel(layers, preinit_weights = None, masks = None,
              initializer = tf.keras.initializers.HeNormal(),
              activation = 'relu',
              BatchNorm = False,
              Dropout = 0.5,
              optimizer = tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
             ):
    '''
    Returns a model for pruning.   

    Parameters:  
        - layers: list of integers, specifying nodes in hidden layers, 
                [layer1, layer2, ..., output]  
        - preinit_weights: ndarray, the initialized weights, default to None,
                None when first initializing  
        - masks: list of ndarray, each element is a mask for all the weights  
        - optimizer: optimizer
        - loss: loss function for the nn
        - metrics: metrics for evaluation  

    Returns:
        - model: tf.keras.Sequential model that can be pruned later  
    '''  

    model = tf.keras.Sequential(name = "ModeltoPrune")
    num_layer = len(layers)

    for i in range(num_layer):
        if masks is None:
            mask = None
        else:
            mask = masks[i]

        model.add(customLinear(layers[i], 
        activation=activation,
        initializer=initializer,
        BatchNorm=BatchNorm,
        Dropout=Dropout,
        mask = mask,
        preinit_weights = preinit_weights))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model
    






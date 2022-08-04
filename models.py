'''Initialize custom model for training and pruning.'''

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds  



class maskInit(tf.keras.initializers.Initializer):

  def __init__(self, mask = None, 
               preinit_weights = None,
               initializer= tf.keras.initializers.HeNormal(),
               **kwargs):
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

def makeFC(preinit_weights = None, masks = None,
              layers = [784, 128, 10],
              activation = 'relu',
              BatchNorm = False,
              Dropout = None,
              optimizer = tf.keras.optimizers.Adam(0.001),
              loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
             ):
    '''
    Returns a model for pruning.   

    Parameters:    
        - preinit_weights: ndarray, the initialized weights, default to None,
                None when first initializing  
        - masks: list of ndarray, each element is a mask for all the weights  
        - layers: list of integers, specifying nodes in hidden layers, 
                [layer1, layer2, ..., output]
        - activation: string, activation function for hidden layers
        - BatchNorm: boolean, whether to use batch normalization
        - Dropout: list of floats, dropout rate
        - optimizer: optimizer
        - loss_fn: loss function for the nn
        - metrics: metrics for evaluation  

    Returns:
        - model: tf.keras.Sequential model that can be pruned later  
    '''  

    model = tf.keras.Sequential(name = "ModeltoPrune")
    model.add(tf.keras.layers.InputLayer(input_shape = layers[0]))
    num_layer = len(layers)

    if BatchNorm and Dropout is None:
        for i in range(num_layer):
            if masks is None:
                mask = None
            else:
                # the masks in pruning function includes the biases
                # which are np.ones(shape of bias)
                mask = masks[2*i]
            if preinit_weights is None:
                preinit_weight = None
            else:
                preinit_weight = preinit_weights[2*i]

            model.add(tf.keras.layers.Dense(layers[i], 
            activation=activation,
            kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))
            model.add(tf.keras.layers.BatchNormalization())

    if BatchNorm and Dropout:
        for i in range(num_layer):
            if masks is None:
                mask = None
            else:
                # the masks in pruning function includes the biases
                # which are np.ones(shape of bias)
                mask = masks[2*i]
            if preinit_weights is None:
                preinit_weight = None
            else:
                preinit_weight = preinit_weights[2*i]

            dropout = Dropout[i]
            model.add(tf.keras.layers.Dense(layers[i], 
            activation=activation,
            kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout))

    if not BatchNorm and Dropout:
        for i in range(num_layer):
            if masks is None:
                mask = None
            else:
                # the masks in pruning function includes the biases
                # which are np.ones(shape of bias)
                mask = masks[2*i]
            if preinit_weights is None:
                preinit_weight = None
            else:
                preinit_weight = preinit_weights[2*i]
                
            dropout = Dropout[i]
            model.add(tf.keras.layers.Dense(layers[i], 
            activation=activation,
            kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))
            model.add(tf.keras.layers.Dropout(dropout))

    if not BatchNorm and Dropout is None:
        for i in range(num_layer):
            if masks is None:
                mask = None
            else:
                # the masks in pruning function includes the biases
                # which are np.ones(shape of bias)
                mask = masks[2*i]
            if preinit_weights is None:
                preinit_weight = None
            else:
                preinit_weight = preinit_weights[2*i]
                
            model.add(tf.keras.layers.Dense(layers[i], 
            activation=activation,
            kernel_initializer=maskInit(mask=mask, preinit_weights = preinit_weight)))

    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=metrics)
    model.summary()

    return model

class customLinear(tf.keras.layers.Layer):  

    def __init__(self, num_out,
                activation = 'relu',
                initializer = tf.keras.initializers.HeNormal(),
                BatchNorm=None, 
                Dropout=None,
                mask=None,
                preinit_weights=None,
                **kwargs):
        '''
        Returns a custom linear layer.

        Parameters:
            - num_out: number of output nodes
            - activation: activation function
            - initializer: initializer of layer, default to kaiming
            - BatchNorm: bool, whether to use batch normalization
            - Dropout: float, dropout rate
            - mask: mask covering weights for pruning
            - preinit_weights: weights before training
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
'''Auxiliary functions.'''

import numpy as np
import tensorflow as tf  

def customPruneFC(model, prune_perc = 0.5):
  '''Returns a mask covering weights of lower magnitudes.''' 
  masks = [] 
  for layer in model.trainable_weights:
    # ignore the bias
    weight = layer.numpy()
    if len(weight.shape)==2:
      perc = np.percentile(weight, prune_perc * 100)
      mask = np.array(weight>perc, dtype = 'float32')
      # print(tf.math.count_nonzero(mask)/len(mask.flatten()))
      masks.append(mask)
    else:
      masks.append(np.ones_like(weight))
  return masks  



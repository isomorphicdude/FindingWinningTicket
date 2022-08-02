'''Prune the fully-connected neural network.'''  

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm  
from models import *
from helper import *


def train_one_step(model, 
                x,y,
                masks, optimizer, 
                loss_fn, 
                train_acc = tf.keras.metrics.
                SparseCategoricalAccuracy(name = 'train_accuracy'),
                train_loss = tf.keras.metrics.Mean(name = 'train_loss'),
                ):
    '''
    Trains the model from one sample.

    Parameters:  
      - model: model to train
      - masks: list of masks,
               length equal to the no. of layers (including the biases
               whose masks are 1)
      - optimizer: tf.keras.optimizer
      - loss_fn: loss function
      - x: image
      - y: label  
    '''
    
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
        
    grads = tape.gradient(loss, model.trainable_variables)
    
    if masks is not None:
      grad_masked = []
      
      # Element-wise multiplication between computed gradients and masks
      for grad, mask in zip(grads, masks):
        grad_masked.append(tf.math.multiply(grad, mask))
      
      optimizer.apply_gradients(zip(grad_masked, model.trainable_variables))

    else:
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_acc(y, y_pred)  


def iterPruning(modelFunc,
                ds_train,
                ds_test,
                model_params,
                epochs = 10,
                num_pruning = 10,
                step_perc = 0.5):  
  '''
  Returns winning ticket and prints train&test accuracies.  

  Args:
    - modelFunc: function for initializing model, 
                 has func(weights, masks)
    - epochs: epochs to train the model before pruning
    - num_pruning: no. of rounds to prune
    - step_perc: percentage to prune
  '''
  masks_set = [None]
  ticket_hist = []
  # init_weight_set = [None]
  # unpack parameters
  optimizer = model_params['optimizer']

  train_loss = model_params['train_loss']
  train_acc = model_params['train_acc']  

  for i in range(0, num_pruning):
    print(f"\n \n Iterative pruning round: {i} \n \n")

    # initialize and train network
    model_to_prune = modelFunc(None, masks_set[i])
    
    # get init weights before training
    pretrained_weights = getInitWeight(model_to_prune)

    # training the model before pruning
    print("\n Start original model training. \n")  

    for epoch in range(epochs):  

      print(f"Epoch {epoch}")
      train_loss.reset_states()
      train_acc.reset_states()  

      for x,y in ds_train:

        train_one_step(model_to_prune, masks_set[i], optimizer, x, y)

    print("\n")
    test_acc = model_to_prune.evaluate(ds_test)[1]
    print(f"Model to prune has acc: {test_acc} before pruning.")
    print("\n")

    # prune and create mask using percentile
    next_masks = customPruneFC(model_to_prune, step_perc)
    masks_set.append(next_masks)

    # initialize the lottery tickets
    re_ticket = modelFunc(pretrained_weights, next_masks)
    numParam(re_ticket)

    # train the lottery tickets  
    print("\n Start Lottery ticket training \n")

    for epoch in range(epochs):

      print(f"Epoch {epoch}")
      train_loss.reset_states()
      train_acc.reset_states()

      for x,y in ds_train:
        train_one_step(re_ticket, next_masks, optimizer, x, y)

    # evaluate acc of lottery tickets on the same set
      print("\n")
      ticket_acc = re_ticket.evaluate(ds_test)[1]
      ticket_hist.append(ticket_acc)
      print(f"Ticket has acc: {ticket_acc} after training.")
      print("\n")

      # print("\n Sanity check")
      # numParam(re_ticket)

      if ticket_acc > test_acc:  
        print(f"Early stop, ticket accuracy: {ticket_acc}")
        break

  return re_ticket, ticket_hist

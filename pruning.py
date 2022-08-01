'''Prune the fully-connected neural network.'''  

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm  


def train_one_step(model, masks, optimizer, 
                loss_fn, 
                train_acc = tf.keras.metrics.
                SparseCategoricalAccuracy(name = 'train_accuracy'),
                train_loss = tf.keras.metrics.Mean(name = 'train_loss'),
                x, y):
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


def iterPruning(epochs = 10, # 5
                num_pruning = 10,
                step_perc = 0.5):  
  '''
  Returns ticket.
  '''
  masks_set = [None]
  init_weight_set = [None]
  ticket_hist = []

  for i in range(0, num_pruning):
    print(f"\n \n Iterative pruning round: {i} \n \n")
    # train network
    model_to_prune = mask_fc(masks_set[i], None)
    
    # get init weights before training
    pretrained_weights = getInitWeight(model_to_prune)

    # training the model before pruning
    print("\n Start original model training. \n")  

    for epoch in range(epochs):  

      print(f"Epoch {epoch}")
      train_loss.reset_states()
      train_accuracy.reset_states()  

      for x,y in ds_train:

        train_one_step(model_to_prune, masks_set[i], tf.keras.optimizers.Adam(0.001), x, y)

    print("\n")
    test_acc = model_to_prune.evaluate(ds_test)[1]
    print(f"Model to prune has acc: {test_acc} before pruning.")
    print("\n")

    # prune and create mask using percentile
    next_masks = customPruneFC(model_to_prune, step_perc)
    masks_set.append(next_masks)

    # initialize the lottery tickets
    print(next_masks[0].shape)
    re_ticket = mask_fc(next_masks, pretrained_weights)
    numParam(re_ticket)

    # train the lottery tickets  
    print("\n Start Lottery ticket training \n")

    for epoch in range(epochs):

      print(f"Epoch {epoch}")
      train_loss.reset_states()
      train_accuracy.reset_states()

      for x,y in ds_train:
        train_one_step(re_ticket, next_masks, tf.keras.optimizers.Adam(0.001), x, y)
    # print(f"\n Epoch {epoch}: Train Loss {train_loss.result()} and Train Accuracy {train_accuracy.result()}")

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

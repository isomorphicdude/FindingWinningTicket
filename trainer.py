'''Prune the fully-connected neural network.'''  

from bitarray import test
import numpy as np
import datetime
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
                train_loss = tf.keras.metrics.Mean(name = 'train_loss')
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


def test_one_step(model, 
              x,y, 
              loss_fn, 
              test_acc = tf.keras.metrics.
              SparseCategoricalAccuracy(name = 'test_accuracy'),
              test_loss = tf.keras.metrics.Mean(name = 'test_loss')
              ):
  
  '''
    Tests the model from one sample.

    Parameters:  
      - model: model to train
      - x,y: sample and label
      - loss_fn: loss function 
  '''  

  prediction = model(x)
  loss = loss_fn(y, x)
  test_loss(loss)
  test_acc(y, prediction)



def iterPruning(modelFunc,
                ds_train,
                ds_test,
                model_params,
                train_params,
                epochs = 10,
                num_pruning = 10,
                step_perc = 0.5):  
  '''
  Returns winning ticket and prints train&test accuracies.  

  Args:
    - modelFunc: function for initializing model, 
                 has func(weights, masks)
    - ds_train:
    - ds_test:
    - model_params:
    - train_params:
    - epochs: epochs to train the model before pruning
    - num_pruning: no. of rounds to prune
    - step_perc: percentage to prune
  '''  
  
  masks_set = [None]
  original_acc = []
  # ticket_hist = []
  # init_weight_set = [None]

  # unpack parameters
  optimizer = model_params['optimizer']
  loss_fn = model_params['loss_fn']
  train_loss = train_params['train_loss'](name = 'train_og_l')
  train_acc = train_params['train_acc'](name = 'train_og_a')
  test_loss = train_params['train_loss'](name = 'test_og_l')
  test_acc = train_params['train_acc'](name = 'test_og_a')
  

  # to use tensorboard
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
  train_og_log_dir = 'logs/' + current_time + '/train_og'
  test_og_log_dir = 'logs/' + current_time + '/test_og'  

  train_og = tf.summary.create_file_writer(train_og_log_dir)
  test_og = tf.summary.create_file_writer(test_og_log_dir)

  for i in range(0, num_pruning):
    print(f"\n \n Iterative pruning round: {i} \n \n")

    # initialize and train network
    model_to_prune = modelFunc(None, masks_set[i])
    
    # get init weights before training
    pretrained_weights = getInitWeight(model_to_prune)

    # training the model before pruning
    print("\n Start original model training. \n")  

    for epoch in range(epochs):  

      print(f"Epoch {epoch} for original \n")
      train_loss.reset_states()
      train_acc.reset_states()
      test_loss.reset_states()
      test_acc.reset_states()

      for x,y in ds_train:
        train_one_step(model_to_prune, x,y, 
                      masks_set[i], optimizer,
                      loss_fn, train_acc, train_loss)

      with train_og.as_default():
        tf.summary.scalar('train_og_loss', train_loss.result(), step=epoch)
        tf.summary.scalar('train_og_accuracy', train_acc.result(), step=epoch)


      for x,y in ds_test:
        test_one_step(model_to_prune, x, y, loss_fn,
                      train_acc, train_loss)
      with test_og.as_default():
        tf.summary.scalar('test_og_loss', test_loss.result(), step=epoch)
        tf.summary.scalar('test_og_accuracy', test_acc.result(), step=epoch)

      # stores original accuracy
    original_acc = test_acc.result()

    # prune and create mask using percentile
    next_masks = customPruneFC(model_to_prune, step_perc)
    masks_set.append(next_masks)

    # initialize the lottery tickets
    re_ticket = modelFunc(pretrained_weights, next_masks)
    numParam(re_ticket)

    # train the lottery tickets  
    print("\n Start Lottery ticket training \n")

    # instantiate metrics to use tensorboard  
    train_ticket_loss = train_params['train_loss'](name = 'train_ticket_l')
    train_ticket_acc = train_params['train_acc'](name = 'train_ticket_a')
    test_ticket_loss = train_params['train_loss'](name = 'test_ticket_l')
    test_ticket_acc = train_params['train_acc'](name = 'test_ticket_a')

    train_ticket_log_dir = 'logs/' + current_time + '/train_ticket'
    test_ticket_log_dir = 'logs/' + current_time + '/test_ticket'  
    train_ticket = tf.summary.create_file_writer(train_ticket_log_dir)
    test_ticket = tf.summary.create_file_writer(test_ticket_log_dir)  

    

    for epoch in range(epochs):
      # train the same number of epochs for lottery tickets
      print(f"Epoch {epoch} for lottery ticket \n")
      train_ticket_loss.reset_states()
      train_ticket_acc.reset_states()
      test_ticket_loss.reset_states()
      test_ticket_acc.reset_states()

      for x,y in ds_train:
        # in batches, train the lottery tickets
        train_one_step(re_ticket, x,y, 
                      next_masks, optimizer,
                      loss_fn, train_acc, train_loss)
      with train_ticket.as_default():
        tf.summary.scalar('train_ticket_loss', train_ticket_loss.result(), step=epoch)
        tf.summary.scalar('train_ticket_accuracy', train_ticket_acc.result(), step=epoch)

    # evaluate acc of lottery tickets on the same set  
      for x,y in ds_test:
        test_one_step(re_ticket, x, y, loss_fn,
                      test_ticket_acc, test_ticket_loss)
      with test_ticket.as_default():
        tf.summary.scalar('test_ticket_loss', test_ticket_loss.result(), step=epoch)
        tf.summary.scalar('test_ticket_accuracy', test_ticket_acc.result(), step=epoch)

      ticket_acc = test_ticket_acc.result()


      if ticket_acc > original_acc:  
        print(f"\n Early stop, ticket accuracy: {ticket_acc} \n")
        re_ticket.save(f'saved_models/ticket_acc_{ticket_acc}'+f'pruning round_{i}')
        # new_model = tf.keras.models.load_model('saved_model/my_model')
        # to reload the model
        break
    re_ticket.save(f'saved_models/ticket_acc_{test_ticket_acc.result()}'+f' pruning round_{i}')

  

  

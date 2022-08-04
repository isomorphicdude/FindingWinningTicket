'''Implements pruning class.'''  

import numpy as np
import tensorflow as tf
from models import *
from helper import *
from trainer import iterPruning


class pruning(object):

    '''Sets up a pruning experiment.'''  

    def __init__(self, ds_train, ds_test, 
                model_params,
                train_params,
                epochs_for_pruning = 10,
                num_pruning = 10,
                step_perc = 0.5):
        '''
        Args:   
            - ds_train: dataset for training, already batched
            - ds_test: dataset for testing, already batched
            - model_params: dictionary of parameters
                            contains the following entries  
                            - 'layers'
                            - 'initializer'
                            - 'activation'
                            - 'BatchNorm'
                            - 'Dropout'
                            - 'optimizer'
                            - 'loss_fn'
                            - 'metrics'
            - train_params: 
            - epochs: epochs to train the model before pruning
            - num_pruning: no. of rounds to prune
            - step_perc: percentage to prune
        '''
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.model_params = model_params
        self.train_params = train_params
        self.epochs_for_pruning = epochs_for_pruning
        self.num_pruning = num_pruning
        self.step_perc = step_perc    

    @property
    def batch_size(self):
        for img,label in self.ds_train:
            num = img.shape[0]
        return num

    # TODO: add Dataset implementaion
    
    # @property
    # def dset(self):
        # some processing goes in here
        # to modify user input
        # pass
    # @dset.setter

    # @property
    # def ds_train(self):
    #     pass

    # @property
    # def ds_test(self):
    #     pass
    

    def makeModel(self, preinit_weights = None, masks = None):
        '''
        Initializes model before pruning.
        '''
        return makeFC(
                    preinit_weights,
                    masks,
                    **self.model_params)
    
    def prune(self):
        '''
        Iteratively prunes the model.
        '''  
        iterPruning(self.makeModel,
                    self.ds_train,
                    self.ds_test,
                    self.model_params,
                    self.train_params,
                    epochs = self.epochs_for_pruning,
                    num_pruning=self.num_pruning,
                    step_perc=self.step_perc)  

    def test_run(self):
        '''Fits the model for testing.'''  
        model = self.makeModel()
        model.compile(self.model_params['optimizer'],
                    self.model_params['loss_fn'],
                    self.model_params['metrics'])
        model.fit(self.ds_train,
                batch_size = self.batch_size,
                epochs = self.epochs_for_pruning,
                validation_data = self.ds_test)



        


    
    
    
    



'''Implements pruning class.'''  

import numpy as np
import tensorflow as tf
from models import *
from helper import *
from trainer import iterPruning


class pruning(object):

    '''Sets up a pruning experiment.'''  

    def __init__(self, dset, model_params,
                epochs_for_pruning = 10,
                num_pruning = 10,
                step_perc = 0.5):
        '''
        Args:  
            - dset: dataset
            - model_params: dictionary of parameters
            - epochs: epochs to train the model before pruning
            - num_pruning: no. of rounds to prune
            - step_perc: percentage to prune
        '''
        self.dset = dset
        self.model_params = model_params
        self.epochs_for_pruning = epochs_for_pruning
        self.num_pruning = num_pruning
        self.step_perc = step_perc    

    # TODO: add Dataset implementaion
    
    @property
    def dset(self):
        # some processing goes in here
        # to modify user input
        pass
    # @dset.setter

    @property
    def ds_train(self):
        pass

    @property
    def ds_test(self):
        pass
    

    def makeModel(self, preinit_weights = None, masks = None):
        '''
        Initializes model before pruning.
        '''
        return makeFC(
                    preinit_weights,
                    masks,
                    self.model_params)
    
    def prune(self):
        '''
        Prunes the model.
        '''  
        iterPruning(self.makeModel,
                    self.ds_train,
                    self.ds_test,
                    self.model_params,
                    epochs = self.epochs_for_pruning,
                    num_pruning=self.num_pruning,
                    step_perc=self.step_perc)



        


    
    
    
    



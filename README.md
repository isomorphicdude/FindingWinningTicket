# Finding the Winning Ticket in a Neural Network  

## Background  

In the seminal paper *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks* by Jonathan Frankle and Michael Carbin, the lottery ticket hypothesis was proposed.    

The hypothesis asserts that any neural network contains a sub-network that can reach e test accuracy of the original network after training independently for at most the same number of iterations [[1]](#1).  

## Implementation  

To find this sub-network, the authors have suggested iterative magnitude pruning. Namely, we follow the following procedure  

1. Randomly initialize the neural network $f(\mathbf{x};\mathbf{\theta_0})$, where the parameters $\mathbf{\theta_0}$ follows an initial distribution $D_0$

2. Train the network for $j$ iterations to get parameters $\mathbf{\theta_j}$  

3. Remove $p^{1/n}\%$ of the weights, recording the weights removed by a binary mask $m$

4.  Reset remaining parameters to the corresponding ones in $\mathbf{\theta_0}$ and train this sub-network $f(\mathbf{x};\mathbf{\theta_0} \odot m)$  

5.  Repeat the above steps for $n$ rounds.  

Only fully-connected neural networks are implemented at this stage. We use the same idea in this implementation. 

## Usage  

An example is given in the file `experiment.ipynb`. You may create a `pruning` instance and specify various parameters, including whether to include dropout layers and batch normalization.  

An application of this project can be found [here](https://github.com/Julie3399/UROP-2022).  







## References
<a id="1">[1]</a>
Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. arXiv: Learning.


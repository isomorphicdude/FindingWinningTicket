'''Gets MNIST from LeCunn's website.'''  

import requests
import argparse

base_url = 'http://yann.lecun.com/exdb/mnist/'  

str_list = ['train-images-idx3-ubyte.gz', 
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz']

for str in str_list:
    response = requests.get(base_url+str)
    open(str, 'wb').write(response.content)

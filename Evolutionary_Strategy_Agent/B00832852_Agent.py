import random
from copy import deepcopy
from tqdm.auto import tqdm
import time
from two_d_set import two_d_set

import numpy as np

NUM_POINTS =12

class MLP_M:

    def __init__(self, input , hiddenL1 , hiddenL2, output):
        # setting MLP parameters bias and weights for three hidden layers
        self.b1=np.array(0.5)
        self.b2=np.array(0.5)
        self.b3=np.array(0.5)
        # Three weights
        # 1st hidden layer with input -weight1
        # 1st hidden layer and 2nd hidden layer -weight2
        # 2nd hidden with output -weight3
        self.w1=np.random.rand(hiddenL1,input)-0.5
        self.w2=np.random.rand(hiddenL2,hiddenL1)-0.5
        self.w3=np.random.rand(output,hiddenL2)-0.5

    def preprocessing(self,data):
        # Create an array of data and preprocess by subtracting the mean and dividing by sd
        data=np.array(data)
        return ((data-data.mean())/data.std()).reshape(28,1)

    def predict(self,data):
        data=self.preprocessing(data)
        #Computes weights and add bias for each layer
        #Add the sigmoid activation function for each layer
        layer1_in=np.matmul(self.w1,data)+self.b1
        layer1_in=1/(1 + np.exp(-layer1_in))
        layer2_in=np.matmul(self.w2,layer1_in)+self.b2
        layer2_in=1/(1 + np.exp(-layer2_in))
        layer3_in=np.matmul(self.w3,layer2_in)+self.b3
        layer3_in=1/(1 + np.exp(-layer3_in))
        return float(layer3_in)

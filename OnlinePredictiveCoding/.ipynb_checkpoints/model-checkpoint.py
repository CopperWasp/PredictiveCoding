import numpy as np
import pandas as pd
import copy

num_layers = 3


def quant(x, l):  # l: num_layers, x:input
    one_hot = copy.deepcopy(x)
    one_hot[one_hot != 0] = 1
    step = (x - one_hot) / (l-1)
    x_list = []
    
    for i in range(l):  # top down
        x_list.append(one_hot + i * step)
    
    return x_list


class error_module:
    def __init__(self, size, lr):
        self.size = size
        self.w = np.zeros(size)
        self.lr = lr

    def predict(self, x):
        return np.dot(self.w, x)

    def update(self, x, y):
        yhat = self.predict(x)  # regression
        loss = 0.5 * (y - yhat)**2
        self.w += self.lr * (y - yhat)
        return loss

    def reset(self):
        self.w = np.zeros(self.size)



class classifier_module:
    def __init__(self, size, lr):
        self.size = size
        self.w = np.zeros(size)
        self.lr = lr

    def predict(self, x):
        return np.dot(self.w, x)

    def update(self, x, y):
        loss = np.maximum(0, 1.0 - y * np.dot(self.w, x))
        if loss > 0: self.w += x * y * self.lr
        return loss

    def reset(self):
        self.w = np.zeros(self.size)



class oco_classifier:
    def __init__(self, size, C):
        self.size = size
        self.w = np.zeros(size)
        self.C = C

    def predict(self, x):
        return np.dot(self.w, x)

    def update(self, x, y):
        loss = np.maximum(0, 1.0 - y * np.dot(self.w, x))
        if loss > 0:
            self.w += np.minimum(self.C, loss/(np.square(np.linalg.norm(x))+1e-6)) * x * y
        return loss

    def reset(self):
        self.w = np.zeros(self.size)
        
        
# predictive coding model trial
class opc:
    def __init__(self, in_size, lr):
        self.in_size = in_size
        self.num_layers = num_layers
        self.w = []
        self.lr = lr
    
        for i in range(self.num_layers): # top-down
            self.w.append(np.zeros(in_size))
        
        
    def predict(self, x, return_sum=True):
        x_list = quant(x, self.num_layers)
        yhat_list = []
        
        for i in range(self.num_layers):
            yhat = np.dot(self.w[i], x_list[i])
            yhat_list.append(yhat)
            
        if return_sum: return np.sum(np.array(yhat_list))
        return yhat_list
            
        
    
    
    def update(self, x, y):
        loss_list = np.zeros(self.num_layers)
        yhat_list = self.predict(x, return_sum=False)
        
        # bottom up, calculate losses
        loss_list[-1] = np.maximum(0, 1.0 - y * yhat_list[self.num_layers-1])
        self.w[self.num_layers-1] += x * y * self.lr
        
        if loss_list[-1] == 0:
            return 0
        
        for i in range(self.num_layers-2, -1, -1):  # starts from the bottom error layer
            label = loss_list[i+1]
            prediction = yhat_list[i]
            loss_list[i] = 0.5 * (label - prediction)**2
            self.w[i] += self.lr * (label - prediction)
            
        return np.mean(loss_list)
    
    
    def reset(self):
        self.w = []
        for i in range(self.num_layers): # top-down
            self.w.append(np.zeros(self.in_size))

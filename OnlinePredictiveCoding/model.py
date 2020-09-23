import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.functional as F
from torch.autograd  import Variable
import torch.optim as optim


class error_module(nn.Module):
    def __init__(self,size):
        super(error_module,self).__init__()
        self.error_linear = nn.Linear(size,1)
        self.Var_e = Variable(torch.ones(1, 1), requires_grad=True)
    def forward(self,x,prev_error):
        x = self.error_linear(x) + self.Var_e * prev_error

        return x



class classifier_module(nn.Module):
    def __init__(self,size):
        super(classifier_module,self).__init__()
        self.classifier_linear = nn.Linear(size,1)
        self.Var_w = Variable(torch.ones(1, 1), requires_grad=True)
    def forward(self,x, prev_error):
        x = self.classifier_linear(x) + self.Var_w * prev_error

        return x


class OPNet(nn.Module):
    def __init__(self,number_layers,size):
        super(OPNet,self).__init__()
        self.classifier_module = classifier_module(size)
        self.number_layers = number_layers
        self.error_modules = nn.ModuleList([error_module(size) for i in range(number_layers-1)])

    def forward(self,x):
        predict= torch.zeros(1, 1).double()
        errors = []
        errors.append(torch.zeros(1, 1).double())
        for i in range (self.number_layers - 1):
            predict = self.error_modules[i](x[i], predict)
            errors.append(predict - errors[-1])


        pred = self.classifier_module(x[-1], predict)
        errors.append(pred - errors[-1])

        return pred, errors

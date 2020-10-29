import numpy as np
import pandas as pd
from torch import nn
import copy
import torch.optim as optim
import torch
from torch.autograd  import Variable



def quant2(x, l):  # l: num_layers, x:input
    one_hot = copy.deepcopy(x)
    one_hot[one_hot != 0] = 1
    step = (x - one_hot) / float(l-1)
    x_list = []
    
    for i in range(l):  # top down
        x_list.append(one_hot + i * step)
    
    return x_list


def quant(x, l):  # l: num_layers, x:input
    one_hot = copy.deepcopy(x)
    one_hot[one_hot != 0] = 1
    step = (np.square(x) - one_hot / float(l-1))
    x_list = []
    
    for i in range(l):  # top down
        if i == l-1:
            x_list.append(x)
        else:
            x_list.append((one_hot + i * step))
    
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
            self.w += np.minimum(self.C, loss/np.square(np.linalg.norm(x)+1e-5)) * x * y
        return loss

    def reset(self):
        self.w = np.zeros(self.size)
        
        
# predictive coding model trial
class opc:
    def __init__(self, in_size, lr, num_layers):
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
            
            
class opc_backprop(nn.Module):
    def __init__(self, in_size, lr, num_layers):
        super(opc_backprop, self).__init__()
        
        self.lr = lr
        self.in_size = in_size
        self.num_layers = num_layers
        self.w = nn.ModuleList([])       
        for i in range(num_layers):
            self.w.append(nn.Linear(in_size, 1, bias=True))  # these biases are extremely important for the performance
        
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

 
    def predict(self, x, return_sum=True):
        return self.forward(x)
  
            
    def forward(self, x):
        X_list, out = quant(x, self.num_layers), 0
        
        for i in range(len(X_list)):
            x_current = torch.FloatTensor(X_list[i])
            out += self.w[i](x_current)
        
        return out
        
    
    def update(self, x, y):
        self.optimizer.zero_grad()
        output = self.predict(x)
        
        if np.sign(output.detach().numpy()) == y:
            return 0
        
        loss = self.criterion(output, torch.FloatTensor([y]))    
        loss.backward()
        self.optimizer.step()
        return loss
        
        
    def reset(self):
        self.w = nn.ModuleList([])
        for i in range(self.num_layers):
            self.w.append(nn.Linear(self.in_size, 1))
            

    def criterion(self, y, yhat):
        zero = torch.Tensor([0])
        return torch.sum(torch.max(zero, 1.0 - y * yhat))
            
            
            
        
        
        
############# Jeev's implementation ################
class error_module(nn.Module):
    def __init__(self,size):
        super(error_module,self).__init__()
        self.error_linear = nn.Linear(size,1, bias=True)
        self.Var_e = 1 #Variable(torch.ones(1, 1), requires_grad=True)
    def forward(self,x,prev_error):
        x = self.error_linear(x) + self.Var_e * prev_error
        
        return x



class classifier_module(nn.Module):
    def __init__(self,size):
        super(classifier_module,self).__init__()
        self.classifier_linear = nn.Linear(size,1, bias=True)
        self.Var_w = 1 #Variable(torch.ones(1, 1), requires_grad=True)
    def forward(self,x, prev_error):
        x = self.classifier_linear(x) +  self.Var_w * prev_error 
        
        return x
    

class MyHingeLoss(torch.nn.Module):

    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, output, target):

        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss
        

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
            errors.append(torch.norm(predict - errors[-1]))
        
            
        pred = self.classifier_module(x[-1], predict) 
        errors.append(torch.norm(pred - errors[-1]))
        
        return pred, errors

class opcbackprop:
    def __init__(self,in_size,lr, num_layers):
        self.in_size = in_size
        self.number_layers = num_layers
        self.lr = lr
        self.model = OPNet(num_layers,in_size).to(torch.double)
        self.criterion = MyHingeLoss()
        self.optim = optim.SGD(self.model.parameters(), lr=lr)
        self.w = -1
        
    def predict(self, x, return_sum = True, detach=True):
        x_list = quant(x, self.number_layers)
        x_tensor = torch.from_numpy(np.array(x_list))
        yhat = self.model(x_tensor)
        if detach:
            return yhat[0].detach()
        else:
            return yhat[0]
             
    def update(self,x,y):
        y = torch.from_numpy(y.reshape(1,1))
        pred = self.predict(x, detach=False)
        loss = self.criterion(pred,y)
        if torch.sign(pred).detach().numpy()[0][0]!= y:
            loss.backward()
            self.optim.step()
            
        return loss
            
    def reset(self):
        self.model = OPNet(self.number_layers,self.in_size).to(torch.double)
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr)

    

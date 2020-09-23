import numpy as np
import pandas as pd


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
            self.w += np.minimum(self.C, loss/(np.square(np.linalg.norm(x))+ 1e-6)) * x * y
        return loss

    def reset(self):
        self.w = np.zeros(self.size)


class online_predictive_coding:
    def __init__(self, size, lr):
        

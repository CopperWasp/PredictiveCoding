import pandas as pd
import numpy as np


root_path, extension = "./datasets/", "_numeric"

def get_path(name):
    '''returns a path pair to the preprocessed datasets
    X and y csv files.'''
    path = root_path + name + extension
    return path + "_X.csv", path + "_y.csv"


def read_dataset(X_path, y_path):
    '''reads and returns numpy arrays in a given pair of paths for
    X and y.'''
    X = pd.read_csv(X_path).values
    y = pd.read_csv(y_path)['0'].values
    return X, y


def simulate_varying(X, cov_strength):  # multivariate normal distribution
    '''Get the data and generate a varying feature space pattern.
    Possible concerns: thresholding messing up the distribution?'''

    # create a covariance matrix
    num_features = len(X[0])
    cov = np.random.rand(num_features, num_features) + cov_strength
    cov = np.dot(cov, cov.transpose())  # to have a positive semi-definite matrix

    # create a mean vector
    mean = np.random.rand(len(X[0]))

    # sample from multivariate gaussian w/ given mean and cov
    spaces = np.random.multivariate_normal(mean, cov, len(X))

    # threshold samples for 1-hot encoding
    spaces[spaces < 0] = 0
    spaces[spaces != 0] = 1

    return spaces

def simulate_random_varying(X, cov_strength=0): # discrete uniform distribution
    matrix = np.random.randint(2, size=(len(X), len(X[0])))
    return matrix


def simulate_nothing(X, cov_strength=0):
    return np.ones_like(X)


def quant(x, l):  # l: num_layers, x:input
    one_hot = []
    for i in x:
        if i != 0:
            one_hot.append(1)
        else:
            one_hot.append(0)
    one_hot = np.array(one_hot)

    qt = (one_hot-x)/l
    qts = []
    qts.append(one_hot)

    for i in range(l):
        qts.append(x + qt * (l-i+1))

    qts.append(x)

    return np.array(qts)

from sklearn.utils import shuffle
import copy
import parameters as p
import torch
import numpy as np

def train(X, y, model):
    losses, predictions = [], []
    error_count = 0

    for i in range(len(X)):
        yhat = model.predict(X[i])
        loss = model.update(X[i], y[i])
        predictions.append(yhat)
        losses.append(loss)

        if torch.is_tensor(yhat):
            yhat = yhat.detach().numpy()
        
        if np.sign(yhat) != y[i]:
            error_count += 1

    return losses, predictions, error_count/len(X), model.w



def cross_validation(X, y, model, num_folds, scenario_function):
    fold_errors = []
    fold_losses = []
    fold_weights = []
    fold_masks = []

    for i in range(num_folds):
        X, y = shuffle(X, y)
        X_copy = copy.deepcopy(X)
        fold_mask = scenario_function(X_copy, p.cov_strength)
        X_copy *= fold_mask
        fold_masks.append(fold_mask)
        model.reset()

        losses, predictions, error_rate, weights = train(X_copy, y, model)
        #print(error_rate)

        fold_errors.append(error_rate)
        fold_losses.append(losses)
        fold_weights.append(weights)

    return fold_errors, fold_losses, fold_weights, fold_masks

from sklearn.utils import shuffle
import copy
import parameters as p
import numpy as np

def train(X, y, model):
    losses, predictions = [], []
    error_count = 0
    cum_error_rate = []
    instances = []
    p = int(0.1 * len(X))
    for i in range(len(X)):
        yhat = model.predict(X[i])
        loss = model.update(X[i], y[i])
        predictions.append(yhat)
        #print(type(loss))
        
        losses.append(loss)
        
        if np.sign(yhat) != y[i]:
            error_count += 1
        if i%p == 0:
            cum_error_rate.append(error_count/(i+1))
            instances.append(i)
        

    return losses, cum_error_rate, instances, predictions, error_count/len(X), model.w



def cross_validation(X, y, model, num_folds, scenario_function,cov_strength):
    fold_errors = []
    fold_losses = []
    fold_weights = []
    fold_masks = []
    fold_cum_error_rate =[]

    for i in range(num_folds):
        X, y = shuffle(X, y)
        X_copy = copy.deepcopy(X)
        fold_mask = scenario_function(X_copy, cov_strength)
        X_copy *= fold_mask
        fold_masks.append(fold_mask)
        model.reset()

        losses, cum_error_rate, instances, predictions, error_rate, weights = train(X_copy, y, model)
        #print(error_rate)
        #print(np.mean(losses))
        #print('fold: {}'.format(i))
        fold_errors.append(error_rate)
        fold_losses.append(losses)
        fold_weights.append(weights)
        fold_cum_error_rate.append(cum_error_rate)

    return fold_errors, fold_cum_error_rate, instances, fold_losses, fold_weights, fold_masks

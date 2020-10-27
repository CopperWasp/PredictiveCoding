import dataloader as dl
import model

random_state = 1
dataset_name = 'german'
folds = 20
learning_rate = 0.01
scenario = 'varying_gaussian'
model_type = 'opc'
num_layers = [3]
cov_strength = {
             'varying_gaussian + 0.5': 0.5}

scenarios = {'varying_gaussian + 0.5': dl.simulate_varying}



datasets = [ 'german', 'ionosphere', 'spambase','svmguide3','wpbc','wdbc','magic04','a8a']

models = {
    'hinge_gradient': model.classifier_module,
    'se_gradient': model.error_module,
    'hinge_oco': model.oco_classifier,
    'opc': model.opc,
    'opcbackprop':model.opcbackprop}

import dataloader as dl
import model

random_state = 1
dataset_name = 'german'
folds = 20
learning_rate = 0.01
scenario = 'varying_gaussian'
model_type = 'opc'
num_layers = 3
cov_strength = 0.5

scenarios = {'full': dl.simulate_nothing,
             'varying_gaussian': dl.simulate_varying,
             'varying_uniform': dl.simulate_random_varying}



datasets = ['a8a', 'german', 'ionosphere', 'spambase', 'magic04']

models = {
    'hinge_gradient': model.classifier_module,
    'se_gradient': model.error_module,
    'hinge_oco': model.oco_classifier,
    'opc': model.opc,
    'opcbackprop':model.opcbackprop}

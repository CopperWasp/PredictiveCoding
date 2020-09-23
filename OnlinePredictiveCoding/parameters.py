import dataloader as dl
import model

random_state = 1
dataset_name = 'german'
folds = 20
learning_rate = 0.01
scenario = 'varying_gaussian'
model_type = 'hinge_oco'

scenarios = {'full': dl.simulate_nothing,
             'varying_gaussian': dl.simulate_varying,
             'varying_uniform': dl.simulate_random_varying}



datasets = ['german', 'ionosphere', 'spambase', 'a8a', 'svmguide3']

models = {
    'hinge_gradient': model.classifier_module,
    'se_gradient': model.error_module,
    'hinge_oco': model.oco_classifier,
    'predictive_coding': model.online_predictive_coding}

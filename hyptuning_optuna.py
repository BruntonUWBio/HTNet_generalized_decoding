import optuna, joblib, pdb, os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify GPU to use
from run_nn_models import run_nn_models


##################USER-DEFINED PARAMETERS##################
lp = '.../' # xarray data loadpath
sp = '.../'
combined_sbjs = False  # if true, combine data from multiple subjects during training; else do single-subject fits

# Specify number of folds
if combined_sbjs:
    n_folds = 12
else:
    n_folds = 3
D = 2 # not fine-tuning this parameter
##################USER-DEFINED PARAMETERS##################

hyperparams_eegnet = {'dropoutRate': [.2, .8],
                      'kernLength': [24, 136],
                      'F1': [4, 20],
                      'dropoutType': ['Dropout', 'SpatialDropout2D'],
                      'kernLength_sep': [24, 136]}
hyperparams_htnet = {'dropoutRate': [.2, .8],
                              'kernLength': [24, 136],
                              'F1': [4, 20],
                              'dropoutType': ['Dropout', 'SpatialDropout2D'],
                              'kernLength_sep': [24, 136]}
hyperparams_rf = {'n_estimators': [25, 250], 'max_depth': [2, 12]}

# Perform each combination (run CNN models together if parameters are the same)
if hyperparams_eegnet == hyperparams_htnet:
    # Run tuning of both CNN models together
    perform_param_combos(hyperparams_eegnet, sp, n_folds,
                         combined_sbjs, ['eegnet', 'eegnet_hilb'])
else:
    perform_param_combos(hyperparams_eegnet,
                         sp, n_folds, combined_sbjs, ['eegnet'])
    perform_param_combos(hyperparams_htnet,
                         sp, n_folds, combined_sbjs, ['eegnet_hilb'])
perform_param_combos(hyperparams_rf, sp, n_folds,
                     combined_sbjs, ['rf'])




def perform_param_combos(param_dict, sp, n_folds,
                         combined_sbjs, models):
    '''
    Takes in hyperparameter combinations as a dictionary and runs model fitting
    '''
    # Generate all hyperparameter combinations
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    study.set_user_attr('param_dict', param_dict)
    study.set_user_attr('sp', sp)
    study.set_user_attr('n_folds', n_folds)
    study.set_user_attr('combined_sbjs', combined_sbjs)
    study.set_user_attr('models', models)
    if 'rf' in models:
        study.optimize(objective, n_trials=25)
    else:
        study.optimize(objective, n_trials=100)
    joblib.dump(study, sp + 'optuna_study_' + str(models) + '.pkl')


def objective(trial):
    params = trial.study.user_attrs
    modeltype = trial.suggest_categorical('modeltype', params['models'])
    if modeltype == 'rf':
        return run_nn_models(params["sp"], params['n_folds'], params['combined_sbjs'], models=modeltype,
                             test_day='last',
                             n_estimators=trial.suggest_int('n_estimators',
                                                            params['param_dict']['n_estimators'][0],
                                                            params['param_dict']['n_estimators'][1],
                                                            step=5),
                             max_depth=trial.suggest_int('max_depth',
                                                         params['param_dict']['max_depth'][0],
                                                         params['param_dict']['max_depth'][1]))
    else:
        F1 = trial.suggest_int("F1", params['param_dict']['F1'][0], params['param_dict']['F1'][1])
        return run_nn_models(params["sp"], params['n_folds'], params['combined_sbjs'], models=modeltype,
                             test_day='last',
                             dropoutRate=trial.suggest_float("dropoutRate",
                                                             params['param_dict']['dropoutRate'][0],
                                                             params['param_dict']['dropoutRate'][1]),
                             kernLength=trial.suggest_int("kernLength",
                                                          params['param_dict']['kernLength'][0],
                                                          params['param_dict']['kernLength'][1],
                                                          step=8),
                             F1=F1, D=D, F2=F1*D,
                             dropoutType=trial.suggest_categorical("dropoutType", params['param_dict']['dropoutType']),
                             kernLength_sep=trial.suggest_int("kernLength_sep",
                                                              params['param_dict']['kernLength_sep'][0],
                                                              params['param_dict']['kernLength_sep'][1],
                                                              step=8))
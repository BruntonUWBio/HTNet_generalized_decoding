'''
Train and fine-tune decoders. Should work with other datasets as long as they are
in the same xarray format (will need to specify loadpath too).
'''
import numpy as np
import pdb,os,time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #specify GPU to use
from run_nn_models import run_nn_models
from transfer_learn_nn import transfer_learn_nn
from model_utils import unseen_modality_test, diff_specs
from transfer_learn_nn_eeg import transfer_learn_nn_eeg

t_start = time.time()
##################USER-DEFINED PARAMETERS##################
data_lp = '.../' # data load path

# Where data will be saved: rootpath + dataset + '/'
rootpath = '.../'
dataset = 'move_rest_ecog'

### Tailored decoder params (within participant) ###
n_folds_tail = 3 # number of folds (per participant)
spec_meas_tail = ['power'] # 'power', 'power_log', 'relative_power', 'phase', 'freqslide'
hyps_tail = {'F1' : 20, 'dropoutRate' : 0.693, 'kernLength' : 64,
             'kernLength_sep' : 56, 'dropoutType' : 'SpatialDropout2D',
             'D' : 2, 'n_estimators' : 240, 'max_depth' : 9}
hyps_tail['F2'] = hyps_tail['F1'] * hyps_tail['D'] # F2 = F1 * D

### Same modality decoder params (across participants) ###
n_folds_same = 36 # number of total folds
spec_meas_same = ['power'] # 'power', 'power_log', 'relative_power', 'phase', 'freqslide'
hyps_same = {'F1' : 19, 'dropoutRate' : 0.342, 'kernLength' : 24,
             'kernLength_sep' : 88, 'dropoutType' : 'Dropout',
             'D' : 2, 'n_estimators' : 240, 'max_depth' : 6}
hyps_same['F2'] = hyps_same['F1'] * hyps_same['D'] # F2 = F1 * D

### Unseen modality testing params (across participants) ###
eeg_lp = '.../' # path to EEG xarray data
eeg_roi_proj_lp = '.../' # path to EEG projection matrix

### Fine-tune same modality decoders ###
model_type_finetune = 'eegnet_hilb' # NN model type to fine-tune (must be either 'eegnet_hilb' or 'eegnet')
layers_to_finetune = ['all',['conv2d','batch_normalization'],
                      ['batch_normalization','depthwise_conv2d','batch_normalization_1'],
                      ['separable_conv2d','batch_normalization_2']]
# Options:  'all' - allow entire model to be retrained  
#           ['conv2d','batch_normalization']
#           ['batch_normalization','depthwise_conv2d','batch_normalization_1']
#           ['separable_conv2d','batch_normalization_2']
#           None - transfer learning of new last 3 layers
sp_finetune = [rootpath + dataset + '/tf_all_per/',
               rootpath + dataset + '/tf_per_1dconv/',
               rootpath + dataset + '/tf_depth_per/',
               rootpath + dataset + '/tf_sep_per/',
               rootpath + dataset + '/tf_single_sub/'] # where to save output (should match layers_to_finetune)

# How much train/val data to use, either by number of trials or percentage of available data
use_per_vals = True #if True, use percentage values (otherwise, use number of trials)
per_train_trials = [.17,.33,.5,0.67]
per_val_trials = [.08,.17,.25,0.33]
n_train_trials = [16,34,66,100]
n_val_trials = [8,16,34,50]

### Train same modality decoders with different numbers of training participants ###
max_train_parts = 10 # use 1--max_train_subs training participants
n_val_parts = 1 # number of validation participants to use
##################USER-DEFINED PARAMETERS##################


#### Tailored decoder training ####
for s,val in enumerate(spec_meas_tail):
    do_log = True if val == 'power_log' else False
    compute_val = 'power' if val == 'power_log' else val
    single_sp = rootpath + dataset + '/single_sbjs_' + val + '/'
    combined_sbjs = False
    if not os.path.exists(single_sp):
        os.mkdirs(single_sp)
    if s==0:
        models = ['eegnet_hilb','eegnet','rf','riemann'] # fit all decoder types
    else:
        models = ['eegnet_hilb'] # avoid fitting non-HTNet models again
    run_nn_models(single_sp, n_folds_tail, combined_sbjs, lp=data_lp, test_day = 'last', do_log=do_log,
                  epochs=300, patience=30, models=models, compute_val=compute_val,
                  F1 = hyps_tail['F1'], dropoutRate = hyps_tail['dropoutRate'], kernLength = hyps_tail['kernLength'],
                  kernLength_sep = hyps_tail['kernLength_sep'], dropoutType = hyps_tail['dropoutType'],
                  D = hyps_tail['D'], F2 = hyps_tail['F2'], n_estimators = hyps_tail['n_estimators'], max_depth = hyps_tail['max_depth'])


#### Same modality training ####
for s,val in enumerate(spec_meas_same):
    do_log = True if val == 'power_log' else False
    compute_val = 'power' if val == 'power_log' else val
    multi_sp = rootpath + dataset  + '/combined_sbjs_' + val + '/'
    if not os.path.exists(multi_sp):
        os.mkdirs(multi_sp)
    combined_sbjs = True
    if s==0:
        models = ['eegnet_hilb','eegnet','rf','riemann'] # fit all decoder types
    else:
        models = ['eegnet_hilb'] # avoid fitting non-HTNet models again
    run_nn_models(multi_sp, n_folds_same, combined_sbjs, lp=data_lp, test_day = 'last', do_log=do_log,
                  epochs=300, patience=20, models=models, compute_val=compute_val,
                  F1 = hyps_same['F1'], dropoutRate = hyps_same['dropoutRate'], kernLength = hyps_same['kernLength'],
                  kernLength_sep = hyps_same['kernLength_sep'], dropoutType = hyps_same['dropoutType'],
                  D = hyps_same['D'], F2 = hyps_same['F2'], n_estimators = hyps_same['n_estimators'], max_depth = hyps_same['max_depth'])

#### Unseen modality testing ####
for s,val in enumerate(spec_meas_same):
    if s==0:
        models = ['eegnet_hilb','eegnet','rf','riemann'] # fit all decoder types
    else:
        models = ['eegnet_hilb']
    for mod_curr in models:
        unseen_modality_test(eeg_lp, eeg_roi_proj_lp, rootpath + dataset + '/',
                             pow_type = val, model_type = mod_curr)
    

#### Same modality fine-tuning ####
spec_meas = 'power'
for j,curr_layer in enumerate(layers_to_finetune):
    # Create save directory if does not exist already
    if not os.path.exists(sp_finetune[j]):
        os.makedirs(sp_finetune[j])

    # Fine-tune with each amount of train/val data
    if curr_layer==layers_to_finetune[-1]:
        single_sub = True
    else:
        single_sub = False
    
    lp_finetune = rootpath + dataset  + '/combined_sbjs_'+spec_meas+'/'
    if use_per_vals:
        for i in range(len(per_train_trials)):
            transfer_learn_nn(lp_finetune, sp_finetune[j], eeg_lp,
                              model_type = model_type_finetune, layers_to_finetune = curr_layer,
                              use_per_vals = use_per_vals, per_train_trials = per_train_trials[i], 
                              per_val_trials = per_val_trials[i],single_sub = single_sub, epochs=300, patience=20) 
    else:
        for i in range(len(n_train_trials)):
            transfer_learn_nn(lp_finetune, sp_finetune[j], eeg_lp,
                              model_type = model_type_finetune, layers_to_finetune = curr_layer,
                              use_per_vals = use_per_vals, n_train_trials = n_train_trials[i],
                              n_val_trials = n_val_trials[i], single_sub = single_sub, epochs=300, patience=20)
            
#### Unseen modality fine-tuning ####
spec_meas = 'relative_power'
for j,curr_layer in enumerate(layers_to_finetune):
    # Create save directory if does not exist already
    if not os.path.exists(sp_finetune[j]):
        os.makedirs(sp_finetune[j])

    # Fine-tune with each amount of train/val data
    if curr_layer==layers_to_finetune[-1]:
        single_sub = True
    else:
        single_sub = False
    
    lp_finetune = rootpath + dataset  + '/combined_sbjs_'+spec_meas+'/'
    if use_per_vals:
        for i in range(len(per_train_trials)):
            transfer_learn_nn_eeg(lp_finetune, sp_finetune[j][:-1]+'_eeg/',
                                  model_type = model_type_finetune, layers_to_finetune = curr_layer,
                                  use_per_vals = use_per_vals, per_train_trials = per_train_trials[i], 
                                  per_val_trials = per_val_trials[i],single_sub = single_sub, epochs=300, patience=20) 
    else:
        for i in range(len(n_train_trials)):
            transfer_learn_nn_eeg(lp_finetune, sp_finetune[j][:-1]+'_eeg/',
                                  model_type = model_type_finetune, layers_to_finetune = curr_layer,
                                  use_per_vals = use_per_vals, n_train_trials = n_train_trials[i],
                                  n_val_trials = n_val_trials[i], single_sub = single_sub, epochs=300, patience=20)


#### Training same modality decoders with different numbers of training participants ####
for i in range(max_train_parts):
    sp_curr = rootpath + dataset + '/combined_sbjs_ntra'+str(i+1)+'/'
    combined_sbjs = True
    if not os.path.exists(sp_curr):
        os.mkdirs(sp_curr)
    run_nn_models(sp_curr,n_folds_same,combined_sbjs,test_day = 'last', do_log=False,
                  epochs=300, patience=20, models=['eegnet_hilb','eegnet','rf','riemann'], compute_val='power',
                  n_val = n_val_parts, n_train = i + 1, F1 = hyps_same['F1'], dropoutRate = hyps_same['dropoutRate'],
                  kernLength = hyps_same['kernLength'], kernLength_sep = hyps_same['kernLength_sep'], dropoutType = hyps_same['dropoutType'],
                  D = hyps_same['D'], F2 = hyps_same['F2'], n_estimators = hyps_same['n_estimators'], max_depth = hyps_same['max_depth'])

#### Pre-compute difference spectrograms for ECoG and EEG datasets ####
diff_specs(rootpath + dataset  + '/combined_sbjs/', data_lp, ecog = True)
diff_specs(rootpath + dataset  + '/combined_sbjs/', eeg_lp, ecog = False)
    
print('Elapsed time: '+str(time.time() - t_start))
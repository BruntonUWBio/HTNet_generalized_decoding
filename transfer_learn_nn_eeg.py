'''
Perform transfer learning on trained (multi-subject) NN model using various amounts of data.
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os,glob,natsort,pdb,pickle,time,sys
import pandas as pd
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    #Choose GPU 0 as a default if not specified
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Input, Flatten, Dense, Activation
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from keras import backend as K
from tqdm import tqdm

# Custom imports
from htnet_model import htnet
from model_utils import load_data, folds_choose_subjects, subject_data_inds, get_custom_motor_rois, proj_mats_good_rois

def compute_accs(model_in, sbj_order_train, x_train, y_train, sbj_order_val, 
                 x_val, y_val, sbj_order_test, x_test, y_test, proj_mat_out):
    '''
    Compute train/val/test accuracies given data and model.
    '''
    acc_lst = []
    preds = model_in.predict([x_train,proj_mat_out[sbj_order_train,...]]).argmax(axis = -1) 
    acc_lst.append(np.mean(preds == y_train.argmax(axis=-1)))
    preds = model_in.predict([x_val,proj_mat_out[sbj_order_val,...]]).argmax(axis=-1)
    acc_lst.append(np.mean(preds == y_val.argmax(axis=-1)))
    preds = model_in.predict([x_test,proj_mat_out[sbj_order_test,...]]).argmax(axis = -1)
    acc_lst.append(np.mean(preds == y_test.argmax(axis=-1)))
    
    return np.asarray(acc_lst)

def run_transfer_learning(model_fname, sbj_order_train, x_train, y_train, 
                          sbj_order_val, x_val, y_val, sbj_order_test, x_test, y_test,
                          proj_mat_out, chckpt_path, layers_to_finetune = None, norm_rate = 0.25,
                          loss='categorical_crossentropy', optimizer='adam',
                          patience = 5, early_stop_monitor = 'val_loss', do_log=False, nb_classes = 2,
                          epochs=20):
    '''
    Perform NN model fitting for transfer learning. Based heavily on code written by @zsteineh.
    '''
    # Load pre-trained model
    pretrain_model = tf.keras.models.load_model(model_fname)
    
    # Compute pre-trained model accuracies (if test_day = 'last', should match previously saved accuracy files).
    acc_pretrain = compute_accs(pretrain_model, sbj_order_train, x_train, y_train, sbj_order_val, 
                                x_val, y_val, sbj_order_test, x_test, y_test, proj_mat_out)
#     print('Test accuracy:', acc_pretrain[2])
    
    # Set up pre-trained model for transfer learning
    if layers_to_finetune is None:
        # If no layers to finetune, then freeze the last three layers and retrain
        x = pretrain_model.layers[-4].output
        x = Flatten(name = 'flatten2')(x)
        x = Dense(nb_classes, name = 'dense', kernel_constraint = max_norm(norm_rate))(x)
        softmax = Activation('softmax', name = 'softmax')(x)

        transfer_model = Model(inputs=[pretrain_model.input[0], pretrain_model.input[1]], outputs=softmax)
        
        # Set only last 3 layers to be trainable
        for l in transfer_model.layers:
            l.trainable = False
        for l in transfer_model.layers[-3:]:
            l.trainable = True #train last 3 layers
    elif layers_to_finetune == ['all']:
        # Allow all layers to be trained
        transfer_model = pretrain_model
        for l in transfer_model.layers:
            l.trainable = True
    else:
        # Finetune specific layers from layers_to_finetune
        transfer_model = pretrain_model
        for l in transfer_model.layers:
            l.trainable = False # ensure all layers start as being not trainable
        
        for layer in layers_to_finetune:
            transfer_model.get_layer(layer).trainable = True #set specified layers to be trained
    
    # Set up comiler, checkpointer, and early stopping during model fitting
    transfer_model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
    checkpointer = ModelCheckpoint(filepath=chckpt_path,verbose=1,save_best_only=True)
    early_stop = EarlyStopping(monitor=early_stop_monitor, mode='min',
                               patience=patience, verbose=0) #stop if val_loss doesn't improve after certain # of epochs

    # Perform model fitting using Keras
    t_start_fit = time.time()
    h = transfer_model.fit([x_train,proj_mat_out[sbj_order_train,...]], y_train, batch_size = 16, epochs = epochs, 
                        verbose = 2, validation_data=([x_val,proj_mat_out[sbj_order_val,...]], y_val),
                        callbacks=[checkpointer,early_stop])
    t_fit_total = time.time() - t_start_fit
    # Find out when training stopped
    last_epoch = len(h.history['loss'])
    if last_epoch<epochs:
        last_epoch -= patience # revert to epoch where best model was found
#     last_epoch = early_stop.stopped_epoch
#     if (last_epoch == 0):
#         last_epoch = 20
#     else: 
#         last_epoch += 1
    
    # Load model weights and check accuracy of fit model to train/val/test data
    transfer_model.load_weights(chckpt_path)
    acc_trained = compute_accs(transfer_model, sbj_order_train, x_train, y_train, sbj_order_val, 
                                x_val, y_val, sbj_order_test, x_test, y_test, proj_mat_out)
    tf.keras.backend.clear_session() # avoids slowdowns when running fits for many folds
    
    return acc_pretrain, acc_trained, np.array([last_epoch,t_fit_total])

def run_single_sub_percent_compare(sbj_order_train, x_train, y_train, 
                          sbj_order_val, x_val, y_val, sbj_order_test, x_test, y_test,
                          chckpt_path, norm_rate = 0.25,
                          loss='categorical_crossentropy', optimizer='adam',
                          patience = 5, early_stop_monitor = 'val_loss', do_log=False, nb_classes = 2, 
                          compute_val='power', ecog_srate=500, epochs=20,
                          dropoutRate = 0.25, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                          dropoutType = 'Dropout',kernLength_sep = 16):
       
    ss_model = htnet(nb_classes, Chans = x_train.shape[2],  Samples = x_train.shape[-1], useHilbert=True,
                     dropoutRate = dropoutRate, kernLength = kernLength, F1 = F1, D = D, F2 = F2, 
                     dropoutType = dropoutType,kernLength_sep = kernLength_sep,
                     projectROIs=False,do_log=do_log,
                     compute_val=compute_val,data_srate=ecog_srate)
    
    ss_model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
    
    # Compute initial accuracy (should be near 50%)
    accs_lst_0 = []
    preds_0 = ss_model.predict(x_train).argmax(axis = -1)
    accs_lst_0.append(np.mean(preds_0 == y_train.argmax(axis=-1)))
    preds_0 = ss_model.predict(x_val).argmax(axis = -1)
    accs_lst_0.append(np.mean(preds_0 == y_val.argmax(axis=-1)))
    preds_0 = ss_model.predict(x_test).argmax(axis = -1)  
    accs_lst_0.append(np.mean(preds_0 == y_test.argmax(axis=-1)))
    
    checkpointer = ModelCheckpoint(filepath=chckpt_path,verbose=1,save_best_only=True)
    early_stop = EarlyStopping(monitor=early_stop_monitor, mode='min',
                               patience=patience, verbose=0)
    t_start_fit = time.time()
    h = ss_model.fit(x_train, y_train, batch_size = 16, epochs = epochs, 
                                verbose = 2, validation_data=(x_val, y_val),
                                callbacks=[checkpointer,early_stop])
    t_fit_total = time.time() - t_start_fit
    
    # Find out when training stopped
    last_epoch = len(h.history['loss'])
    if last_epoch<epochs:
        last_epoch -= patience # revert to epoch where best model was found
#     last_epoch = early_stop.stopped_epoch
#     if (last_epoch == 0):
#         last_epoch = 20
#     else: 
#         last_epoch += 1
        
    ss_model.load_weights(chckpt_path)
    accs_lst = []
    preds = ss_model.predict(x_train).argmax(axis = -1)
    accs_lst.append(np.mean(preds == y_train.argmax(axis=-1)))
    preds = ss_model.predict(x_val).argmax(axis = -1)
    accs_lst.append(np.mean(preds == y_val.argmax(axis=-1)))
    preds = ss_model.predict(x_test).argmax(axis = -1)  
    accs_lst.append(np.mean(preds == y_test.argmax(axis=-1)))
    
    tf.keras.backend.clear_session() # avoids slowdowns when running fits for many folds
    
    return accs_lst, np.array([last_epoch,t_fit_total]), accs_lst_0


def transfer_learn_nn_eeg(lp, sp, eeg_data_lp,
                          model_type = 'eegnet_hilb', layers_to_finetune = None,
                          n_train_trials = 50, per_train_trials = 0.6, n_val_trials = 50, per_val_trials = 0.3,
                          n_test_trials = 50, use_per_vals = False,loss='categorical_crossentropy', optimizer='adam',
                          patience=5,early_stop_monitor='val_loss',norm_rate=0.25,use_prev_opt_early_params=True,
                          single_sub=False, compute_val='power', ecog_srate=500, epochs = 20):
    '''
    Main script for performing transfer learning across folds. Matches code from run_nn_models.py.
    
    If doing test_day = 'last', only need to specify train and val trials/percent because test set is known.
    '''
    # Parameters for projection matrix
    custom_rois = True
    n_chans_eeg = 61
    n_chans_ecog = 126 # number of channels in ecog data (expected by model)
    per_test_trials = 0.2 # percentage of EEG data to use for test set
    
    # Ensure layers_to_finetune is a list
    if (layers_to_finetune is not None) and (not isinstance(layers_to_finetune, list)):
        layers_to_finetune = [layers_to_finetune]
    
    # Create suffix for saving files (so can save results from different train/val sizes to same folder)
    if use_per_vals:
        suffix_trials = '_ptra'+str(int(per_train_trials*100))+'_pval'+str(int(per_val_trials*100))
    else:
        suffix_trials = '_ntra'+str(n_train_trials)+'_nval'+str(n_val_trials)+'_ntes'+str(n_test_trials)
    
    # Load param file from pre-trained model
    file_pkl = open(lp+'param_file.pkl', 'rb')
    params_dict = pickle.load(file_pkl)
    file_pkl.close()
    
    # Extract appropriate parameters from param file
    tlim = params_dict['tlim']
    test_day = params_dict['test_day']
#     pats_ids_in = params_dict['pats_ids_in']
    pats_ids_in = ['EE'+str(val).zfill(2) for val in np.arange(1,16).tolist()]
    rand_seed = params_dict['rand_seed']
    n_test_sbj = params_dict['n_test']
    n_val_sbj = params_dict['n_val']
    n_folds = params_dict['n_folds']
    save_suffix = params_dict['save_suffix']
    do_log = params_dict['do_log']
#     data_lp = params_dict['lp']
    if 'n_train' in list(params_dict.keys()):
        n_train_sbj = params_dict['n_train']
    else:
        n_train_sbj = 7
    
    if 'epochs' in list(params_dict.keys()):
        epochs = params_dict['epochs']
        compute_val = params_dict['compute_val']
        ecog_srate = params_dict['ecog_srate']
    if use_prev_opt_early_params:
        # Use model fitting parameters from pre-trained model
        loss = params_dict['loss']
        optimizer = params_dict['optimizer']
        patience = params_dict['patience']
        early_stop_monitor = params_dict['early_stop_monitor']
    
    # Load in hyperparameters
    dropoutRate = params_dict['dropoutRate']
    kernLength = params_dict['kernLength']
    F1 = params_dict['F1']
    D = params_dict['D']
    F2 = params_dict['F2']
    dropoutType = params_dict['dropoutType']
    kernLength_sep = params_dict['kernLength_sep']
    
    # Find pathnames of models from all folds
    model_fnames = natsort.natsorted(glob.glob(lp + 'checkpoint_gen_'+model_type+'_fold*.h5'))
    
    # Set random seed
    np.random.seed(rand_seed)
    
    # Load projection matrix (electrodes to ROI's) for EEG data
    if custom_rois:
        custom_roi_inds = get_custom_motor_rois() # load custom roi's from precentral, postcentral, and inf parietal (AAL2)
    else:
        custom_roi_inds = None
    print("Determining ROIs")
    proj_mat_out,good_ROIs,chan_ind_vals_all = proj_mats_good_rois(['EE01_bH'],
                                                                   n_chans_all = n_chans_eeg,
                                                                   rem_bad_chans=False,
                                                                   dipole_dens_thresh=None,
                                                                   custom_roi_inds=custom_roi_inds,
                                                                   chan_cut_thres=n_chans_eeg,
                                                                   roi_proj_loadpath= eeg_data_lp+'proj_mat/')
    nROIs = len(good_ROIs)
    print("ROIs found")
    n_chans_all = n_chans_eeg

    # Load EEG data for each subject and fit model
    X_all,y_all,_,_,sbj_order_all,_ = load_data(pats_ids_in, eeg_data_lp, test_day=None, tlim=tlim, n_chans_all=n_chans_eeg)
    X_all[np.isnan(X_all)] = 0 # set all NaN's to 0
    
    for pat_ind,curr_pat in enumerate(pats_ids_in):
        # Identify the number of unique labels (or classes) present
        nb_classes = len(np.unique(y_all))

        # Determine train/val/test inds for every fold
        labels_unique = np.unique(y_all)
        nb_classes = len(labels_unique)
        half_n_evs_test = 'nopad' #avoids duplicating events (will take all available events)

        acc_pretrain = np.zeros([n_folds,3])
        acc_trained = acc_pretrain.copy()
        acc_single_sub = acc_pretrain.copy()
        acc_single_sub_0 = acc_single_sub.copy()
        last_epochs_TL = np.zeros([n_folds,2])
        last_epochs_SS = np.zeros([n_folds,2])
        for i in range(n_folds):
            # First, find indices for all events associated with test subject
            other_inds = subject_data_inds(np.full(1, pat_ind), sbj_order_all, labels_unique, i, 
                                           'test_inds', half_n_evs_test, y_all, sp, n_folds, [])
            trainval_inds = np.asarray(list(set(other_inds)))

            # If test_day is not last, then determine number of train, val, and test events (trials) to obtain
            if use_per_vals:
                n_train = int(len(other_inds) * per_train_trials*(1-per_test_trials))
                n_val = int(len(other_inds) * per_val_trials*(1-per_test_trials))
                n_test = int(len(other_inds) * per_test_trials) #(1-per_train_trials-per_val_trials))
            else:
                n_train = int(n_train_trials)
                n_val = int(n_val_trials)
                n_test = int(n_test_trials)

            # Find train event indices
            test_inds_tmp = subject_data_inds(np.full(1, pat_ind), sbj_order_all[trainval_inds], labels_unique, i, 
                                               'test_inds', n_test//nb_classes, y_all[trainval_inds], sp, n_folds, [])
            
            test_inds = trainval_inds[test_inds_tmp] # convert back to original inds

            # Remove train events and choose val inds from remaining events
            # Note: if n_train is larger than available events for training data, finding validation events
            # will throw an error because there are no remaining events to choose from
            valtest_inds = np.asarray(list(set(other_inds) - set(test_inds))) #remove train inds
            if len(valtest_inds) == 0:
                sys.exit("Error: No data to pick from for validation and test sets!")
            val_inds_tmp = subject_data_inds(np.full(1, pat_ind), sbj_order_all[valtest_inds], labels_unique, i, 
                                             'val_inds', n_val//nb_classes, y_all[valtest_inds], sp, n_folds, [])
            val_inds = valtest_inds[val_inds_tmp] # convert back to original inds

            # Remove val events and choose test inds from remaining events
            # Note: if n_train+n_val is larger than available events for training data, finding test events
            # will throw an error because there are no remaining events to choose from
            remain_inds = np.asarray(list(set(valtest_inds) - set(val_inds))) # remove train inds
            if len(remain_inds) == 0:
                sys.exit("Error: No data to pick from for test set!")
            train_inds_tmp = subject_data_inds(np.full(1, pat_ind), sbj_order_all[remain_inds], labels_unique, i, 
                                             'train_inds', n_train//nb_classes, y_all[remain_inds], sp, n_folds, [])
            train_inds = remain_inds[train_inds_tmp] # convert back to original inds

            # Append train/val/test event indices for each fold
    #         train_inds_folds.append(train_inds)
    #         val_inds_folds.append(val_inds)
    #         test_inds_folds.append(test_inds)

            # Reformat data size for NN
            Y_all = np_utils.to_categorical(y_all-1)
            X_all_tmp = np.expand_dims(X_all,1)
            proj_mat_out2 = np.tile(proj_mat_out,[X_all_tmp.shape[0],1,1])
            proj_mat_out2 = np.expand_dims(proj_mat_out2,1)

            # Pad channel dimension to match ECoG data
            X_all_sh = list(X_all_tmp.shape)
            X_all_sh[2] = n_chans_ecog
            X_all_resh = np.zeros(X_all_sh)
            X_all_resh[...,:n_chans_eeg,:] = X_all_tmp
            proj_mat_out3 = np.zeros(list(proj_mat_out2.shape[:-1])+[n_chans_ecog])
            proj_mat_out3[...,:n_chans_eeg] = proj_mat_out2

            # Generate train/val/test data based on event indices for each fold
            X_train = X_all_resh[train_inds,...]
            Y_train = Y_all[train_inds,...]
            sbj_order_train = np.zeros(len(train_inds)).astype('int') # important for projection matrix input
            X_validate = X_all_resh[val_inds,...]
            Y_validate = Y_all[val_inds,...]
            sbj_order_validate = np.zeros(len(val_inds)).astype('int')  # important for projection matrix input
            
            X_test = X_all_resh[test_inds,...]
            Y_test = Y_all[test_inds,...]
            sbj_order_test = np.zeros(len(test_inds)).astype('int')  # important for projection matrix input
            # Note that sbj_order doesn't matter here because all EEG subjects have same electrode locations
            
            # Run transfer learning
            str_len = len('checkpoint_gen_')
            curr_mod_fname = model_fnames[i].split('/')[-1][:-3]
            chckpt_path = sp+'checkpoint_gen_tf_'+curr_pat+'_'+curr_mod_fname[str_len:]+suffix_trials+'.h5'
            acc_pretrain_tmp, acc_trained_tmp, last_epoch_tmp = run_transfer_learning(model_fnames[i], sbj_order_train,
                                                                      X_train, Y_train, sbj_order_validate, 
                                                                      X_validate, Y_validate, sbj_order_test,
                                                                      X_test, Y_test,proj_mat_out3, chckpt_path,
                                                                      layers_to_finetune = layers_to_finetune,
                                                                      norm_rate = norm_rate, loss=loss,
                                                                      optimizer=optimizer, patience = patience,
                                                                      early_stop_monitor = early_stop_monitor,
                                                                      do_log=do_log, nb_classes = nb_classes,
                                                                      epochs = epochs)

            # Here need to run the single subject on the same amount of training and val data
            if single_sub:
                chckpt_path = sp+'checkpoint_gen_tf_single_sub_'+curr_pat+'_'+suffix_trials+'.h5'
                acc_single_sub_tmp, last_epoch_single_tmp, acc_single_sub_tmp_0 = run_single_sub_percent_compare(sbj_order_train, X_train, Y_train, 
                                              sbj_order_validate, X_validate, Y_validate, sbj_order_test, 
                                              X_test, Y_test, chckpt_path, norm_rate = norm_rate,
                                              loss=loss, optimizer=optimizer, patience = patience, 
                                              early_stop_monitor = early_stop_monitor, do_log=do_log, nb_classes = nb_classes,
                                              compute_val=compute_val, ecog_srate=ecog_srate, epochs = epochs,
                                              dropoutRate = dropoutRate, kernLength = kernLength, F1 = F1, D = D, F2 = F2, 
                                          dropoutType = dropoutType,kernLength_sep = kernLength_sep)
                acc_single_sub[i,:] = acc_single_sub_tmp
                last_epochs_SS[i,:] = last_epoch_single_tmp
                acc_single_sub_0[i,:] = acc_single_sub_tmp_0

            # Save train/val/test accuracies for every fold
            acc_pretrain[i,:] = acc_pretrain_tmp
            acc_trained[i,:] = acc_trained_tmp    
            last_epochs_TL[i,:] = last_epoch_tmp
        
        
        # Save accuracies across all folds (adds suffix for number/percentage of trials)
        np.save(sp+'acc_gen_tf_pretrain_'+curr_pat+'_'+model_type+'_'+str(n_folds)+save_suffix+\
                suffix_trials+'.npy',acc_pretrain)
        np.save(sp+'acc_gen_tf_trained_'+curr_pat+'_'+model_type+'_'+str(n_folds)+\
                save_suffix+suffix_trials+'.npy',acc_trained)
        np.save(sp+'last_training_epoch_gen_tf'+curr_pat+'_'+model_type+'_'+str(n_folds)+\
                save_suffix+suffix_trials+'.npy', last_epochs_TL)
        if single_sub:
            np.save(sp+'acc_gen_tf_singlesub_'+curr_pat+'_'+model_type+'_'+str(n_folds)+\
                    save_suffix+suffix_trials+'.npy',acc_single_sub)
            np.save(sp+'acc_gen_tf_singlesub0_'+curr_pat+'_'+model_type+'_'+str(n_folds)+\
                    save_suffix+'.npy',acc_single_sub_0)
            np.save(sp+'last_training_epoch_gen_tf_singlesub_'+curr_pat+'_'+model_type+'_'+str(n_folds)
                    +save_suffix+suffix_trials+'.npy', last_epochs_SS)
        

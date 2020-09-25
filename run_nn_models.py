import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import os,pdb,argparse,pickle,time
import pandas as pd
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    #Choose GPU 0 as a default if not specified (can set this in Python script that calls this)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import backend as K

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from os import path
from itertools import product
import pyriemann

# Custom imports
from htnet_model import htnet
from model_utils import load_data, folds_choose_subjects, subject_data_inds, roi_proj_rf, str2bool

def cnn_model(X_train, Y_train,X_validate, Y_validate,X_test,Y_test,chckpt_path,modeltype,
              proj_mat_out=None,sbj_order_train=None,sbj_order_validate=None,
              sbj_order_test=None,nROIs=100,nb_classes = 2,dropoutRate = 0.25,
              kernLength = 32, F1 = 8, D = 2, F2 = 16, dropoutType = 'Dropout',
              kernLength_sep = 16,loss='categorical_crossentropy',optimizer='adam',
              patience = 5, early_stop_monitor = 'val_loss',do_log=False,epochs=20,
              compute_val='power',ecog_srate=500):
    '''
    Perform NN model fitting based on specified prarameters.
    '''
    # Logic to determine how to run model
    projectROIs = True if proj_mat_out is not None else False #True if there are multiple subjects in train data
    useHilbert = True if modeltype == 'eegnet_hilb' else False #True if want to use Hilbert transform layer
    
    # Load NN model
    model = htnet(nb_classes, Chans = X_train.shape[2], Samples = X_train.shape[-1], 
                  dropoutRate = dropoutRate, kernLength = kernLength, F1 = F1, D = D, F2 = F2, 
                  dropoutType = dropoutType,kernLength_sep = kernLength_sep,
                  ROIs = nROIs,useHilbert=useHilbert,projectROIs=projectROIs,do_log=do_log,
                  compute_val=compute_val,ecog_srate=ecog_srate)
    
    # Set up comiler, checkpointer, and early stopping during model fitting
    model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
#     numParams    = model.count_params() # count number of parameters in the model
    checkpointer = ModelCheckpoint(filepath=chckpt_path,verbose=1,save_best_only=True)
    early_stop = EarlyStopping(monitor=early_stop_monitor, mode='min',
                               patience=patience, verbose=0) #stop if val_loss doesn't improve after certain # of epochs

    # Perform model fitting in Keras (model inputs differ depending on whether or not to project to roi's
    t_start_fit = time.time()
    if projectROIs:
        fittedModel = model.fit([X_train,proj_mat_out[sbj_order_train,...]], Y_train, batch_size = 16, epochs = epochs, 
                                verbose = 2, validation_data=([X_validate,proj_mat_out[sbj_order_validate,...]], Y_validate),
                                callbacks=[checkpointer,early_stop])
    else:
        fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = epochs, 
                                verbose = 2, validation_data=(X_validate, Y_validate),
                                callbacks=[checkpointer,early_stop])
    t_fit_total = time.time() - t_start_fit
    
    # Get the last epoch for training
    last_epoch = len(fittedModel.history['loss'])
    if last_epoch<epochs:
        last_epoch -= patience # revert to epoch where best model was found
    print("Last epoch was: ", last_epoch)
    
    # Load model weights from best model and compute train/val/test accuracies
    model.load_weights(chckpt_path)

    accs_lst = []
    if projectROIs:
        preds       = model.predict([X_train,proj_mat_out[sbj_order_train,...]]).argmax(axis = -1)
        accs_lst.append(np.mean(preds == Y_train.argmax(axis=-1)))
        preds       = model.predict([X_validate,proj_mat_out[sbj_order_validate,...]]).argmax(axis = -1)
        accs_lst.append(np.mean(preds == Y_validate.argmax(axis=-1)))
        preds       = model.predict([X_test,proj_mat_out[sbj_order_test,...]]).argmax(axis = -1)  
        accs_lst.append(np.mean(preds == Y_test.argmax(axis=-1)))
    else:
        preds       = model.predict(X_train).argmax(axis = -1)
        accs_lst.append(np.mean(preds == Y_train.argmax(axis=-1)))
        preds       = model.predict(X_validate).argmax(axis = -1)
        accs_lst.append(np.mean(preds == Y_validate.argmax(axis=-1)))
        preds       = model.predict(X_test).argmax(axis = -1)  
        accs_lst.append(np.mean(preds == Y_test.argmax(axis=-1)))
    tf.keras.backend.clear_session() # avoids slowdowns when running fits for many folds
    return accs_lst, np.array([last_epoch,t_fit_total])

def run_nn_models(sp,n_folds,combined_sbjs,lp,
                  pats_ids_in=['EC01','EC02','EC03','EC04','EC05','EC06',
                               'EC07','EC08','EC09','EC10','EC11','EC12'],
                  n_evs_per_sbj=500,test_day=None,tlim=[-1,1],
                  n_chans_all=140,dipole_dens_thresh=0.2,rem_bad_chans=True,
                  models=['eegnet_hilb','eegnet','rf'],save_suffix='',
                  n_estimators=150,max_depth=8,overwrite=True,dropoutRate=0.25,kernLength=32,
                  F1=8, D=2, F2=16, dropoutType='Dropout', kernLength_sep=16,rand_seed=1337,
                  loss='categorical_crossentropy',optimizer='adam',
                  patience = 5, early_stop_monitor = 'val_loss', do_log = False, n_test=1, n_val=4,
                  roi_proj_loadpath = '/data1/users/stepeter/mvmt_init/ROIproj_matlab_smallROIs/',
                  custom_rois = True, n_train = 7, epochs=20, compute_val='power',ecog_srate=500,
                  half_n_evs_test = 'nopad',trim_n_chans=True):
    '''
    Main function that prepares data and aggregates accuracy values from model fitting.
    Note that overwrite variable no longer does anything.
    Also note that ecog_srate is only needed for frequency sliding computation in neural net (if compute_val=='freqslide')
    '''
    # Ensure pats_ids_in and models variables are lists
    if not isinstance(pats_ids_in, list):
        pats_ids_in = [pats_ids_in]
    if not isinstance(models, list):
        models = [models]
    
    # Save pickle file with dictionary of input parameters (useful for reproducible dataset splits and model fitting)
    params_dict = {'sp':sp, 'n_folds':n_folds, 'combined_sbjs':combined_sbjs, 'lp':lp, 'pats_ids_in':pats_ids_in,
                   'n_evs_per_sbj':n_evs_per_sbj, 'test_day':test_day, 'tlim':tlim, 'n_chans_all':n_chans_all,
                   'dipole_dens_thresh':dipole_dens_thresh, 'rem_bad_chans':rem_bad_chans, 'models':models,
                   'save_suffix':save_suffix, 'n_estimators':n_estimators, 'max_depth':max_depth, 'overwrite':overwrite,
                   'dropoutRate':dropoutRate, 'kernLength':kernLength, 'F1':F1, 'D':D, 'F2':F2, 'dropoutType':dropoutType,
                   'kernLength_sep':kernLength_sep, 'rand_seed':rand_seed, 'loss':loss, 'optimizer':optimizer,
                   'patience':patience, 'early_stop_monitor':early_stop_monitor, 'do_log':do_log, 'n_test':n_test,
                   'n_val':n_val,'n_train':n_train, 'epochs': epochs, 'compute_val':compute_val, 'ecog_srate':ecog_srate,'trim_n_chans':trim_n_chans}
    f = open(sp+'param_file.pkl','wb')
    pickle.dump(params_dict,f)
    f.close()
    
    # Set random seed
    np.random.seed(rand_seed)
    
    # Perform different procedures depending on whether or not multiple subjects are being fit together
    if combined_sbjs:
        # For multi-subject fits, obtain projection matrix and good regions of interest for each subject
        if custom_rois:
            custom_roi_inds = get_custom_motor_rois() # load custom roi's from precentral, postcentral, and inf parietal (AAL2)
        else:
            custom_roi_inds = None
        print("Determining ROIs")
        proj_mat_out,good_ROIs,chan_ind_vals_all = proj_mats_good_rois(pats_ids_in,
                                                                       n_chans_all = n_chans_all,
                                                                       rem_bad_chans=rem_bad_chans,
                                                                       dipole_dens_thresh=dipole_dens_thresh,
                                                                       custom_roi_inds=custom_roi_inds,
                                                                       chan_cut_thres=n_chans_all,
                                                                       roi_proj_loadpath=roi_proj_loadpath)
        nROIs = len(good_ROIs)
        print("ROIs found")
        
        # Retain only the electrodes with nonzero data (initially padded because number of electrodes varies across subjects)
        # proj_mat_out : (len(pats_ids_in) x len(good_ROIs) x n_chans_all)
        if trim_n_chans:
            n_chans_all = len(np.nonzero(proj_mat_out.reshape(-1,proj_mat_out.shape[-1]).mean(axis=0))[0])
            proj_mat_out = proj_mat_out[...,:n_chans_all]
        np.save(sp+"proj_mat_out", proj_mat_out)
        
        # Load ECoG data (if test_day is None, then X_test_orig, y_test_orig, and sbj_order_test_load will be empty)
        X,y,X_test_orig,y_test_orig,sbj_order,sbj_order_test_load = load_data(pats_ids_in, lp,
                                                                              n_chans_all=n_chans_all,
                                                                              test_day=test_day, tlim=tlim)
        X[np.isnan(X)] = 0 # set all NaN's to 0
        # Identify the number of unique labels (or classes) present
        nb_classes = len(np.unique(y))
        
        # Choose which subjects for training/validation/testing for every fold (splits are based on random seed)
        sbj_inds_all_train, sbj_inds_all_val, sbj_inds_all_test = folds_choose_subjects(n_folds, pats_ids_in,
                                                                                        n_test=n_test, n_val=n_val,
                                                                                        n_train=n_train)
        
        # Iterate across all model types specified
        labels_unique = np.unique(y)
        if isinstance(n_evs_per_sbj,str):
            half_n_evs = n_evs_per_sbj
        else:
            half_n_evs = n_evs_per_sbj//len(labels_unique)
#         half_n_evs_test = 'nopad' #avoid duplicating events for test set (okay for train/val sets where it is more important to balance trials across subjects)
        train_inds_folds, val_inds_folds, test_inds_folds = [],[],[]
        for k,modeltype in enumerate(models):
            accs = np.zeros([n_folds,3]) # accuracy table for all NN models
            last_epochs = np.zeros([n_folds,2])
            
            # For the number of folds, pick the events to use
            for i in tqdm(range(n_folds)):
                test_sbj = sbj_inds_all_test[i]
                val_sbj = sbj_inds_all_val[i]
                train_sbj = sbj_inds_all_train[i]

                # Only need to determine train/val/test inds for first modeltype used
                if k==0:
                    # Find train/val/test indices (test inds differ depending on if test_day is specified or not)
                    # Note that subject_data_inds will balance number of trials across classes
                    train_inds, val_inds, test_inds = [],[],[]
                    if test_day is None:
                        test_inds = subject_data_inds(np.full(1, test_sbj), sbj_order, labels_unique, i, 
                                                      'test_inds', half_n_evs_test, y, sp, n_folds, test_inds, overwrite)
                    else:
                        test_inds = subject_data_inds(np.full(1, test_sbj), sbj_order_test_load, labels_unique, i, 
                                                      'test_inds', half_n_evs_test, y_test_orig, sp, n_folds, test_inds, overwrite)
                    val_inds = subject_data_inds(val_sbj, sbj_order, labels_unique, i, 
                                                 'val_inds', half_n_evs, y, sp, n_folds, val_inds, overwrite)
                    train_inds = subject_data_inds(train_sbj, sbj_order, labels_unique, i, 
                                                   'train_inds', half_n_evs, y, sp, n_folds, train_inds, overwrite)
                    train_inds_folds.append(train_inds)
                    val_inds_folds.append(val_inds)
                    test_inds_folds.append(test_inds)
                else:
                    train_inds = train_inds_folds[i]
                    val_inds = val_inds_folds[i]
                    test_inds = test_inds_folds[i]
                
                # Now that we have the train/val/test event indices, generate the data for the models
                X_train = X[train_inds,...]
                Y_train = y[train_inds]
                sbj_order_train = sbj_order[train_inds]
                X_validate = X[val_inds,...]
                Y_validate = y[val_inds]
                sbj_order_validate = sbj_order[val_inds]
                if test_day is None:
                    X_test = X[test_inds,...]
                    Y_test = y[test_inds]
                    sbj_order_test = sbj_order[test_inds]
                else:
                    X_test = X_test_orig[test_inds,...]
                    Y_test = y_test_orig[test_inds]
                    sbj_order_test = sbj_order_test_load[test_inds]

                if modeltype == 'rf':
                    # For random forest, project data from electrodes to ROIs in advance
                    X_train_proj = roi_proj_rf(X_train,sbj_order_train,nROIs,proj_mat_out)
                    X_validate_proj = roi_proj_rf(X_validate,sbj_order_validate,nROIs,proj_mat_out)
                    X_test_proj = roi_proj_rf(X_test,sbj_order_test,nROIs,proj_mat_out)

                    # Create Random Forest classifier model
                    model = RandomForestClassifier(n_estimators=n_estimators,
                                                   max_depth=max_depth, 
                                                   class_weight="balanced",
                                                   random_state=rand_seed,
                                                   n_jobs=1,
                                                   oob_score=True)

                    # Fit model and store train/val/test accuracies
                    t_fit_start = time.time()
                    clf = model.fit(X_train_proj, Y_train.ravel())
                    last_epochs[i,1] = time.time() - t_fit_start
                    accs[i,0] = accuracy_score(Y_train.ravel(), clf.predict(X_train_proj))
                    accs[i,1] = accuracy_score(Y_validate.ravel(), clf.predict(X_validate_proj))
                    accs[i,2] = accuracy_score(Y_test.ravel(), clf.predict(X_test_proj))
                    del X_train_proj,X_validate_proj,X_test_proj
                    
                    # Save model
                    chckpt_path = sp+modeltype+'_fold'+str(i)+save_suffix+'.sav'
                    pickle.dump(clf, open(chckpt_path, 'wb'))
                elif modeltype == 'riemann':
                    # Project data from electrodes to ROIs in advance
                    X_train_proj = roi_proj_rf(X_train,sbj_order_train,nROIs,proj_mat_out)
                    X_validate_proj = roi_proj_rf(X_validate,sbj_order_validate,nROIs,proj_mat_out)
                    X_test_proj = roi_proj_rf(X_test,sbj_order_test,nROIs,proj_mat_out)
                    
                    # Reshape into 3 dimensions
                    X_train_proj2 = X_train_proj.reshape((X_train.shape[0],-1,X_train.shape[-1]))
                    X_validate_proj2 = X_validate_proj.reshape((X_validate.shape[0],-1,X_validate.shape[-1]))
                    X_test_proj2 = X_test_proj.reshape((X_test.shape[0],-1,X_test.shape[-1]))
                    
                    # Find any events where std is 0
                    train_inds_bad = np.nonzero(X_train_proj2.std(axis=-1).max(axis=-1)==0)[0]
                    val_inds_bad = np.nonzero(X_validate_proj2.std(axis=-1).max(axis=-1)==0)[0]
                    test_inds_bad = np.nonzero(X_test_proj2.std(axis=-1).max(axis=-1)==0)[0]
                    if not not train_inds_bad.tolist():
                        first_good_ind = np.setdiff1d(np.arange(X_train_proj2.shape[0]),train_inds_bad)[0]
                        X_train_proj2[train_inds_bad,...] = X_train_proj2[(train_inds_bad*0)+first_good_ind,...]
                    if not not val_inds_bad.tolist():
                        first_good_ind = np.setdiff1d(np.arange(X_validate_proj2.shape[0]),val_inds_bad)[0]
                        X_validate_proj2[val_inds_bad,...] = X_validate_proj2[(val_inds_bad*0)+first_good_ind,...]
                    if not not test_inds_bad.tolist():
                        first_good_ind = np.setdiff1d(np.arange(X_test_proj2.shape[0]),test_inds_bad)[0]
                        X_test_proj2[test_inds_bad,...] = X_test_proj2[(test_inds_bad*0)+first_good_ind,...]
                    
                    # Estimate covariances matrices
                    cov_data_train = pyriemann.estimation.Covariances('lwf').fit_transform(X_train_proj2)
                    cov_data_val = pyriemann.estimation.Covariances('lwf').fit_transform(X_validate_proj2)
                    cov_data_test = pyriemann.estimation.Covariances('lwf').fit_transform(X_test_proj2)

                    # Create MDM model
                    mdm = pyriemann.classification.MDM()
                    
                    # Fit model and store train/val/test accuracies
                    t_fit_start = time.time()
                    clf = mdm.fit(cov_data_train, Y_train.ravel())
                    last_epochs[i,1] = time.time() - t_fit_start
                    accs[i,0] = accuracy_score(Y_train.ravel(), clf.predict(cov_data_train))
                    accs[i,1] = accuracy_score(Y_validate.ravel(), clf.predict(cov_data_val))
                    accs[i,2] = accuracy_score(Y_test.ravel(), clf.predict(cov_data_test))
                    del X_train_proj,X_validate_proj,X_test_proj
                    
                    # Save model
                    chckpt_path = sp+modeltype+'_fold'+str(i)+save_suffix+'.sav'
                    pickle.dump(clf, open(chckpt_path, 'wb'))
                else:
                    # Reformat data size for NN fitting
                    Y_train = np_utils.to_categorical(Y_train-1)
                    X_train = np.expand_dims(X_train,1)
                    Y_validate = np_utils.to_categorical(Y_validate-1)
                    X_validate = np.expand_dims(X_validate,1)
                    Y_test = np_utils.to_categorical(Y_test-1)
                    X_test = np.expand_dims(X_test,1)
                    proj_mat_out2 = np.expand_dims(proj_mat_out,1)
                    
                    # Fit NN model using Keras
                    chckpt_path = sp+'checkpoint_gen_'+modeltype+'_fold'+str(i)+save_suffix+'.h5'
                    accs_lst, last_epoch_tmp = cnn_model(X_train, Y_train,X_validate,Y_validate,X_test,Y_test,
                                                         chckpt_path,modeltype,proj_mat_out2,sbj_order_train,
                                                         sbj_order_validate,sbj_order_test,nROIs=nROIs,
                                                         nb_classes = nb_classes,dropoutRate = dropoutRate, kernLength = kernLength, 
                                                         F1 = F1, D = D, F2 = F2, dropoutType = dropoutType,
                                                         kernLength_sep = kernLength_sep,loss=loss,optimizer=optimizer,
                                                         patience = patience, early_stop_monitor = early_stop_monitor, do_log=do_log,
                                                         epochs = epochs, compute_val = compute_val,ecog_srate=ecog_srate)

                    # Store train/val/test accuracies, and last epoch
                    for ss in range(3):
                        accs[i,ss] = accs_lst[ss]
            
                    last_epochs[i,:] = last_epoch_tmp
                
            # Save accuracies for all folds for one type of model
            np.save(sp+'acc_gen_'+modeltype+'_'+str(n_folds)+save_suffix+'.npy',accs)
            np.save(sp+'last_training_epoch_gen_tf'+modeltype+'_'+str(n_folds)+save_suffix+'.npy', last_epochs)
            
        # Returns average validation accuracy for hyperparameter tuning (will be for last model_type only)
        return accs[:,1].mean()
    else:
        # Single subject model fitting
        for pat_id_curr in pats_ids_in:
            # Load ECoG data
            X,y,X_test,y_test,sbj_order,sbj_order_test = load_data(pat_id_curr, lp,
                                                                   n_chans_all=n_chans_all,
                                                                   test_day=test_day, tlim=tlim)
            X[np.isnan(X)] = 0 # set all NaN's to 0
            # Identify the number of unique labels (or classes) present
            nb_classes = len(np.unique(y))
            
            # Randomize event order (random seed facilitates consistency)
            order_inds = np.arange(len(y))
            np.random.shuffle(order_inds)
            X = X[order_inds,...]
            y = y[order_inds]
            order_inds_test = np.arange(len(y_test))
            np.random.shuffle(order_inds_test)
            X_test = X_test[order_inds_test,...]
            y_test = y_test[order_inds_test]
            
            # Iterate across all model types specified
            for modeltype in models:
                # Reformat data based on model
                if modeltype == 'rf':
                    y2 = y.copy()
                    y_test2 = y_test.copy()
                    X2 = X.copy()
                    X_test2 = X_test.copy()
                elif modeltype == 'riemann':
                    y2 = y.copy()
                    y_test2 = y_test.copy()
                    X2 = X.copy()
                    X_test2 = X_test.copy()
                else:
                    y2 = np_utils.to_categorical(y-1)
                    y_test2 = np_utils.to_categorical(y_test-1)
                    X2 = np.expand_dims(X,1)
                    X_test2 = np.expand_dims(X_test,1)

                # Create splits for train/val and fit model
                split_len = X2.shape[0]//n_folds
                accs = np.zeros([n_folds,3])
                last_epochs = np.zeros([n_folds,2])
                for frodo in range(n_folds):
                    val_inds = np.arange(0,split_len)+(frodo*split_len)
                    train_inds = np.setdiff1d(np.arange(X2.shape[0]),val_inds) #take all events not in val set
                    
                    # Split data and labels into train/val sets
                    X_train = X2[train_inds,...]
                    Y_train = y2[train_inds]
                    X_validate = X2[val_inds,...]
                    Y_validate = y2[val_inds]

                    if modeltype == 'rf':
                        # For random forest, combine electrodes and time dimensions
                        X_train_rf = X_train.reshape(X_train.shape[0],-1)
                        X_validate_rf = X_validate.reshape(X_validate.shape[0],-1)
                        X_test2_rf = X_test2.reshape(X_test2.shape[0],-1)
                        
                        # Create random forest model
                        model = RandomForestClassifier(n_estimators=n_estimators,
                                                       max_depth=max_depth, 
                                                       class_weight="balanced",
                                                       random_state=rand_seed,
                                                       n_jobs=1,
                                                       oob_score=True)

                        # Fit model and store accuracies
                        t_fit_start = time.time()
                        clf = model.fit(X_train_rf, Y_train.ravel())
                        last_epochs[frodo,1] = time.time() - t_fit_start
                        accs[frodo,0] = accuracy_score(Y_train.ravel(), clf.predict(X_train_rf))
                        accs[frodo,1] = accuracy_score(Y_validate.ravel(), clf.predict(X_validate_rf))
                        accs[frodo,2] = accuracy_score(y_test2.ravel(), clf.predict(X_test2_rf))
                        
                        # Save model
                        chckpt_path = sp+modeltype+'_'+pat_id_curr[:3]+'_testday_'+\
                                      str(test_day)+'_fold'+str(frodo)+save_suffix+'.sav'
                        pickle.dump(clf, open(chckpt_path, 'wb'))
                    elif modeltype == 'riemann':
                        # Find any events where std is 0
                        train_inds_bad = np.nonzero(X_train.std(axis=-1).max(axis=-1)==0)[0]
                        val_inds_bad = np.nonzero(X_validate.std(axis=-1).max(axis=-1)==0)[0]
                        test_inds_bad = np.nonzero(X_test2.std(axis=-1).max(axis=-1)==0)[0]
                        if not not train_inds_bad.tolist():
                            first_good_ind = np.setdiff1d(np.arange(X_train.shape[0]),train_inds_bad)[0]
                            X_train[train_inds_bad,...] = X_train[(train_inds_bad*0)+first_good_ind,...]
                        if not not val_inds_bad.tolist():
                            first_good_ind = np.setdiff1d(np.arange(X_validate.shape[0]),val_inds_bad)[0]
                            X_validate[val_inds_bad,...] = X_validate[(val_inds_bad*0)+first_good_ind,...]
                        if not not test_inds_bad.tolist():
                            first_good_ind = np.setdiff1d(np.arange(X_test2.shape[0]),test_inds_bad)[0]
                            X_test2[test_inds_bad,...] = X_test2[(test_inds_bad*0)+first_good_ind,...]
                        
                        
                        # Estimate covariances matrices
                        cov_data_train = pyriemann.estimation.Covariances('lwf').fit_transform(X_train)
                        cov_data_val = pyriemann.estimation.Covariances('lwf').fit_transform(X_validate)
                        cov_data_test = pyriemann.estimation.Covariances('lwf').fit_transform(X_test2)

                        # Create MDM model
                        mdm = pyriemann.classification.MDM()

                        # Fit model and store train/val/test accuracies
                        t_fit_start = time.time()
                        clf = mdm.fit(cov_data_train, Y_train.ravel())
                        last_epochs[frodo,1] = time.time() - t_fit_start
                        accs[frodo,0] = accuracy_score(Y_train.ravel(), clf.predict(cov_data_train))
                        accs[frodo,1] = accuracy_score(Y_validate.ravel(), clf.predict(cov_data_val))
                        accs[frodo,2] = accuracy_score(y_test2.ravel(), clf.predict(cov_data_test))
                        
                        # Save model
                        chckpt_path = sp+modeltype+'_'+pat_id_curr[:3]+'_testday_'+\
                                      str(test_day)+'_fold'+str(frodo)+save_suffix+'.sav'
                        pickle.dump(clf, open(chckpt_path, 'wb'))
                    else:
                        # Fit NN model and store accuracies
                        chckpt_path = sp+'checkpoint_'+modeltype+'_'+pat_id_curr[:3]+'_testday_'+\
                                      str(test_day)+'_fold'+str(frodo)+save_suffix+'.h5'
                        accs_lst, last_epoch_tmp = cnn_model(X_train, Y_train,X_validate,
                                                             Y_validate,X_test2,y_test2,chckpt_path,modeltype,
                                                             nb_classes = nb_classes,dropoutRate = dropoutRate,
                                                             kernLength = kernLength, F1 = F1, D = D, F2 = F2,
                                                             dropoutType = dropoutType, kernLength_sep = kernLength_sep,
                                                             loss=loss,optimizer=optimizer,
                                                             patience = patience, early_stop_monitor = early_stop_monitor,do_log=do_log,
                                                             epochs = epochs, compute_val = compute_val,ecog_srate=ecog_srate)

                        for ss in range(3):
                            accs[frodo,ss] = accs_lst[ss]
                        
                        last_epochs[frodo,:] = last_epoch_tmp
                
                # Save accuracies (train/val/test)
                np.save(sp+'acc_'+modeltype+'_'+pat_id_curr[:3]+'_testday_'+str(test_day)+save_suffix+'.npy',accs)
                np.save(sp+'last_training_epoch_gen_tf'+modeltype+'_'+pat_id_curr[:3]+'_testday_'
                        +str(test_day)+save_suffix+'.npy', last_epochs)
        
        # Return validation accuracy for hyperparameter tuning (assumes only 1 model and 1 subject)
        return accs[:,1].mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform classification with various models (uses xarray input)')
    parser.add_argument('-lp','--load_path', required=False,
                        default='/data1/users/stepeter/cnn_hilbert/ecog_data/xarray/',
                        help='Folder to load ECoG epochs from')
    parser.add_argument('-sp','--save_path',required=True,help='Folder to save xarray outputs')
    parser.add_argument('-pid','--patient_ids', type=str, nargs='+', required=False,
                        default=['a0f66459','c95c1e82','cb46fd46','fcb01f7a','ffb52f92','b4ac1726',
                                 'f3b79359','ec761078','f0bbc9a9','abdb496b','ec168864','b45e3f7b'],
                        help='Patients to convert (based on id)')
    parser.add_argument('-crp','--crop_times', type=float, nargs='+', required=False,default=[-1,1],
                        help='Time (in sec) to crop data to')
    parser.add_argument('-fld','--n_folds',required=True,type=int,
                        help='Number of folds to do (per subject if not combined)')
    parser.add_argument('-evs','--n_evs_per_sbj',required=False,type=int,default=500,
                        help='Number of events per subject (combines all event types)')
    parser.add_argument('-csbj','--combined_sbjs',required=True,type=str2bool, nargs='?',
                        help='Classify with multiple subjects (True) or single subjects (False)')
    parser.add_argument('-tdy','--test_day',required=False,type=str,default=None,help='Test day to use (None or ''last'')')
    parser.add_argument('-cha','--n_chans_all',required=False,type=int,default=140,
                        help='Number greater than maximum chan number plan to use')
    parser.add_argument('-dip','--dipole_dens_thresh',required=False,type=float,default=0.2,
                        help='Electrode density threshold for ROI projection')
    parser.add_argument('-rch','--rem_bad_chans',required=False,type=str2bool, nargs='?',default=True,
                        help='Remove channels that have been previously marked as bad')
    parser.add_argument('-mods','--models',required=False, nargs='+', type=str,default=['eegnet_hilb','eegnet','rf'],
                        help='Model types to use')
    parser.add_argument('-sfx','--save_suffix',required=False,type=str,default='',
                        help='Suffix for save files')
    parser.add_argument('-est','--n_estimators',required=False,type=int,default=150,
                        help='Number of estimators (RF parameter)')
    parser.add_argument('-dep','--max_depth',required=False,type=int,default=8,
                        help='Maximum depth (RF parameter)')
    parser.add_argument('-ove','--overwrite',required=False,type=str2bool, nargs='?',default=True,
                        help='If True, overwrite saved intermediate files (selected events, subject orders per fold, etc.)')
    parser.add_argument('-drr','--dropoutRate',required=False,type=float,default=0.25,
                        help='NN dropout rate')
    parser.add_argument('-kln','--kernLength',required=False,type=int,default=32,
                        help='Temporal kernel length in 1st NN layer')
    parser.add_argument('-f1','--F1',required=False,type=int,default=8,
                        help='F1 parameter in NN model')
    parser.add_argument('-d','--D',required=False,type=int,default=2,
                        help='D parameter in NN model')
    parser.add_argument('-f2','--F2',required=False,type=int,default=16,
                        help='F2 parameter in NN model')
    parser.add_argument('-drt','--dropoutType',required=False,type=str,default='Dropout',
                        help='Type of dropout in NN model')
    parser.add_argument('-skln','--kernLength_sep',required=False,type=int,default=16,
                        help='Temporal kernel length in NN separable conv layer')
    parser.add_argument('-rse','--rand_seed',required=False,type=int,default=1337,
                        help='Random seed to initialize np.random')
    parser.add_argument('-los','--loss',required=False,type=str,default='categorical_crossentropy',
                        help='Loss function to use for NN fits')
    parser.add_argument('-opt','--optimizer',required=False,type=str,default='adam',
                        help='Which optimizer to use for NN training')
    parser.add_argument('-pat','--patience',required=False,type=int,default=5,
                        help='Number of consecutive epochs with no improvement before early stop (for NN)')
    parser.add_argument('-esm','--early_stop_monitor',required=False,type=str,default='val_loss',
                        help='Metric used for NN early stopping during training')
    parser.add_argument('-log','--do_log',required=False,type=str2bool, nargs='?',default=False,
                        help='If True, perform log(1+x) on hilbert envelope (hilbert NN only)')
    parser.add_argument('-nte','--n_test',required=False,type=int,default=1,
                        help='Number of test subjects for multi-subject classification')
    parser.add_argument('-nva','--n_val',required=False,type=int,default=4,
                        help='Number of validation subjects for multi-subject classification')
    parser.add_argument('-ntr','--n_train',required=False,type=int,default=7,
                        help='Number of train subjects for multi-subject classification')
    parser.add_argument('-eps','--epochs',required=False,type=int,default=20,
                        help='Number of epochs during neural net fitting')
    args = parser.parse_args()
    
    run_nn_models(args.save_path,args.n_folds,args.combined_sbjs,args.load_path,args.patient_ids,args.n_evs_per_sbj,
                  args.test_day,args.crop_times,args.n_chans_all,args.dipole_dens_thresh,args.rem_bad_chans,
                  args.models,args.save_suffix,args.n_estimators,args.max_depth,args.overwrite,args.dropoutRate,
                  args.kernLength,args.F1,args.D,args.F2,args.dropoutType,args.kernLength_sep,args.rand_seed,
                  args.loss,args.optimizer,args.patience,args.early_stop_monitor,args.do_log,args.n_test,args.n_val,
                  args.n_train,args.epochs)

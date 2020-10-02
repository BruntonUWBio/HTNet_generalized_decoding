"""
Utility functions for running HTNet keras model and handling data.
"""

import pdb, pickle, os, mne, copy, natsort, glob, pyriemann
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pdb
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D
from tensorflow.keras import utils as np_utils
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import xarray as xr
from os import path
from mne.filter import filter_data
from mne.time_frequency import tfr_morlet
from mne import set_log_level
set_log_level(verbose='ERROR')
from functools import reduce

def hilbert_tf(x):
    #, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform in tensorflow.
    The transformation is done along the last axis by default.
    Adapted from scipy: https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L2012-L2120
    Parameters
    ----------
    x : tensor
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.
    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`
    """
    if x.dtype.is_complex:
        raise ValueError("x must be real.")
#     if N is None:
    if tf.__version__[0]=='1':
        N = x.get_shape()[-1].value
    else:
        N = x.get_shape()[-1]
#     if N <= 0:
#         raise ValueError("N must be positive.")
    
    if tf.__version__[0]=='1':
        Xf = tf.spectral.fft(tf.cast(x,dtype=tf.complex64))
    else:
        Xf = tf.signal.fft(tf.cast(x,dtype=tf.complex64))
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if len(x.get_shape().as_list()) > 1:
    #K.ndim(x) > 1:
        ind = [np.newaxis] * len(x.get_shape().as_list()) #K.ndim(x)
        ind[-1] = slice(None)
        h = h[tuple(ind)]
    X_conv = tf.math.multiply(Xf,tf.cast(tf.convert_to_tensor(h),tf.complex64))
    if tf.__version__[0]=='1':
        X_ifft = tf.spectral.ifft(X_conv)
    else:
        X_ifft = tf.signal.ifft(X_conv)
    return X_ifft

def apply_hilbert_tf(x, envelope=False, do_log=False, compute_val='power', data_srate=250):
    """Compute Hilbert transform of signals w/ zero padding in tensorflow.
    Adapted from MNE function
    Parameters
    ----------
    x : tensor, shape (n_times)
        The signal to convert
    n_fft : int
        Size of the FFT to perform, must be at least ``len(x)``.
        The signal will be cut back to original length.
    envelope : bool
        Whether to compute amplitude of the hilbert transform in order
        to return the signal envelope.
    Returns
    -------
    out : array, shape (n_times)
        The hilbert transform of the signal, or the envelope.
    """
#     #Filter data to limit temporal filtering to specific frequencies
#     x = tf.numpy_function(filter_data,[x, 250, 50, 100], Tout=tf.float32)
    
#     from scipy.signal import hilbert
    if tf.__version__[0]=='1':
        n_x = x.get_shape()[-1].value
    else:
        n_x = x.get_shape()[-1]
    #TO DO: remove last timepoint of signal
    hilb_sig = hilbert_tf(x) #hilbert(x, N=n_fft, axis=-1)[..., :n_x] 
    
    if compute_val=='power':
        out = tf.math.abs(hilb_sig)
        if do_log:
            out = tf.math.log1p(out)
    elif compute_val=='phase':
        out = unwrap(angle_custom(hilb_sig)) #tf.math.cos(angle_custom(hilb_sig)) # angle_custom(hilb_sig)
#         tf.print(tf.math.reduce_mean(out))
    elif compute_val=='freqslide':
        ang = angle_custom(hilb_sig) #tf.math.angle(hilb_sig)
        ang = data_srate*diff(unwrap(ang))/(2*np.pi)
        paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, 1]])
        out = tf.pad(ang, paddings, "CONSTANT") # pad time dimension because of difference function
        # TO DO: apply median filter (use tfa.image.median_filter2d)
#     elif envelope:
#         out = tf.math.abs(hilb_sig)
#         if do_log:
#             out = tf.math.log1p(out)
    return out

def angle_custom(X, epsilon=1.0e-12):
    '''
    Custom atan2 computation of angle from @mjlm (ttps://github.com/tensorflow/tensorflow/issues/6095).
    Avoids real and imaginary values from being exactly 0 which led to nan NN weights.
    '''
    zreal = tf.math.real(X)
    zimag = tf.math.imag(X)
    
    # Add a small number to all zeros, to avoid division by zero:
    zreal = tf.where(tf.equal(zreal, 0.0), zreal+epsilon, zreal)
    zimag = tf.where(tf.equal(zimag, 0.0), zimag+epsilon, zimag)

    angle = tf.where(tf.greater(zreal,0.0), tf.atan(zimag/zreal), tf.zeros_like(zreal))
    angle = tf.where(tf.logical_and(tf.less(zreal,0.0),  tf.greater_equal(zimag,0.0)), tf.atan(zimag/zreal) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(zreal,0.0),  tf.less(zimag,0.0)), tf.atan(zimag/zreal) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(zreal,0.0), tf.greater(zimag,0.0)), 0.5*np.pi * tf.ones_like(zreal), angle)
    angle = tf.where(tf.logical_and(tf.equal(zreal,0.0), tf.less(zimag,0.0)), -0.5*np.pi * tf.ones_like(zreal), angle)
    angle = tf.where(tf.logical_and(tf.equal(zreal,0.0), tf.equal(zimag,0.0)), tf.zeros_like(zreal), angle)
    return angle

def unwrap(p, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor. (Author: Mageneta group)
    Args:
    p: Phase tensor.
    discont: Float, size of the cyclic discontinuity.
    axis: Axis of which to unwrap.
    Returns:
    unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
    ddmod = tf.math.mod(dd + np.pi, 2.0 * np.pi) - np.pi
    idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
    ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
    ph_correct = ddmod - dd
    idx = tf.less(tf.abs(dd), discont)
    ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
    ph_cumsum = tf.cumsum(ph_correct, axis=axis)

    shape = p.get_shape().as_list()
#     if shape[0] is None:
#         shape[0] = 1 # take care of initialization issue (SP 5/6/20)
    shape[axis] = 1
    zeros_mat = tf.zeros_like(p, dtype=p.dtype)
    if shape[0] is None:
        ph_cumsum = tf.concat([zeros_mat[:,:shape[1],:shape[2],:shape[3]], ph_cumsum], axis=axis)
    else:
        ph_cumsum = tf.concat([zeros_mat[:shape[0],:shape[1],:shape[2],:shape[3]], ph_cumsum], axis=axis)
#     ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
    return unwrapped

def diff(x, axis=-1):
    """Take the finite difference of a tensor along an axis. (Author: Mageneta group)
    Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.
    Returns:
    d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
    ValueError: Axis out of range for tensor.
    """
    shape = x.get_shape()
    if axis >= len(shape):
        raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                         (axis, len(shape)))

    begin_back = [0 for unused_s in range(len(shape))]
    begin_front = [0 for unused_s in range(len(shape))]
    begin_front[axis] = 1

    size = shape.as_list()
    size[axis] -= 1
#     if size[0] is None:
#         size[0] = 1 # take care of initialization issue (SP 5/6/20)
    slice_front = x[:,:,:,1:] #tf.slice(x, begin_front, size)
    slice_back = x[:,:,:,:-1] #tf.slice(x, begin_back, size)
    d = slice_front - slice_back
    return d


def proj_to_roi(in_vals):
    '''
    Project x from channels to ROI using proj mat.
    Parameters
    ----------
    in_vals is a list of 2 tensors:
    
    x : tensor, shape (batch,filter,chans,time)
        The signal to project
    proj_mat : tensor, shape (batch,roi,chans)
        The projection matrix from channels to ROIs
    '''
    x = in_vals[0]
    proj_mat = in_vals[1]
    shape_x = x.get_shape().as_list()

    #Apply projection matrix separately for each filter in x (slow...)
    output_list = []
    for i in range(shape_x[1]):
        output_list.append(proj_mat[:,0,...] @ x[:,i,...])
    x_out = tf.stack(output_list,axis=1)
    return x_out


def proj_mats_good_rois(patient_ids,dipole_dens_thresh = .1, n_chans_all = 150,
                        roi_proj_loadpath = '.../',
                        atlas = 'none', rem_bad_chans = True, custom_roi_inds=None, chan_cut_thres = None):
    '''
    Loads projection matrix for each subject and determines good ROIs to use
    
    Parameters
    ----------
    dipole_dens_thresh : threshold to use when deciding good ROI's (based on average channel density for each ROI)
    n_chans_all : number of channels to output (should be >= to maximum number of channels across subjects)
    roi_proj_loadpath : where to load projection matrix CSV files
    atlas : ROI projection atlas to use (aal, loni, brodmann, or none)
    rem_bad_chans : whether to remove bad channels from projection step, defined from abnormal SD or IQR across entire day
    '''
    #Find good ROIs first
    df_all = []
    for s,patient in enumerate(patient_ids):
        df = pd.read_csv(roi_proj_loadpath+atlas+'_'+patient+'_elecs2ROI.csv')
        if s==0:
            dipole_densities = df.iloc[0]
        else:
            dipole_densities += df.iloc[0]
        df_all.append(df)

    dipole_densities = dipole_densities/len(patient_ids) 
    if custom_roi_inds is None:
        good_ROIs = np.nonzero(np.asarray(dipole_densities)>dipole_dens_thresh)[0]
    else:
        good_ROIs = custom_roi_inds.copy()
    
    #Now create projection matrix output (patients x roi x chans)
    n_pats = len(patient_ids)
    proj_mat_out = np.zeros([n_pats,len(good_ROIs),n_chans_all])
    chan_ind_vals_all = []
    for s,patient in enumerate(patient_ids):
        df_curr = df_all[s].copy()
        chan_ind_vals = np.nonzero(df_curr.transpose().mean().values!=0)[0][1:]
        chan_ind_vals_all.append(chan_ind_vals)
        if rem_bad_chans:
            # Load param file from pre-trained model
            file_pkl = open(roi_proj_loadpath+'bad_ecog_electrodes.pkl', 'rb')
            bad_elecs_ecog = pickle.load(file_pkl)
            file_pkl.close()
            inds2drop = bad_elecs_ecog[s]
            if chan_cut_thres is not None:
                all_inds = np.arange(df_curr.shape[0])
                inds2drop = np.union1d(inds2drop,all_inds[all_inds>chan_cut_thres])
            df_curr.iloc[inds2drop] = 0
            #Renormalize across ROIs
            sum_vals = df_curr.sum(axis=0).values
            for i in range(len(sum_vals)):
                df_curr.iloc[:,i] = df_curr.iloc[:,i]/sum_vals[i]
        n_chans_curr = len(chan_ind_vals) #df_curr.shape[0]
        tmp_mat = df_curr.values[chan_ind_vals,:]
        proj_mat_out[s,:,:n_chans_curr] = tmp_mat[:,good_ROIs].T
    return proj_mat_out,good_ROIs,chan_ind_vals_all


def load_data(pats_ids_in, lp, n_chans_all=64, test_day=None, tlim=[-1,1], event_types=['rest','move']):
    '''
    Load ECoG data from all subjects and combine (uses xarray variables)
    
    If len(pats_ids_in)>1, the number of electrodes will be padded or cut to match n_chans_all
    If test_day is not None, a variable with test data will be generated for the day specified
        If test_day = 'last', the last day will be set as the test day.
    '''
    if not isinstance(pats_ids_in, list):
        pats_ids_in = [pats_ids_in]
    sbj_order,sbj_order_test = [],[]
    X_test_subj,y_test_subj = [],[] #placeholder vals
        
    #Gather each subjects data, and concatenate all days
    for j in tqdm(range(len(pats_ids_in))):
        pat_curr = pats_ids_in[j]
        ep_data_in = xr.open_dataset(lp+pat_curr+'_ecog_data.nc')
        ep_times = np.asarray(ep_data_in.time)
        time_inds = np.nonzero(np.logical_and(ep_times>=tlim[0],ep_times<=tlim[1]))[0]
        n_ecog_chans = (len(ep_data_in.channels)-1)
        
        if test_day == 'last':
            test_day_curr = np.unique(ep_data_in.events)[-1] #select last day
        else:
            test_day_curr = test_day
        
        if n_chans_all < n_ecog_chans:
            n_chans_curr = n_chans_all
        else:
            n_chans_curr = n_ecog_chans
            
        
        
        days_all_in = np.asarray(ep_data_in.events)
        
        if test_day is None:
            #No test output here
            days_train = np.unique(days_all_in)
            test_day_curr = None
        else:
            days_train = np.unique(days_all_in)[:-1]
            day_test_curr = np.unique(days_all_in)[-1]
            days_test_inds = np.nonzero(days_all_in==day_test_curr)[0]
            
        #Compute indices of days_train in xarray dataset
        days_train_inds = []
        for day_tmp in list(days_train):
            days_train_inds.extend(np.nonzero(days_all_in==day_tmp)[0])
        
        #Extract data and labels
        dat_train = ep_data_in[dict(events=days_train_inds,channels=slice(0,n_chans_curr),
                                    time=time_inds)].to_array().values.squeeze()
        labels_train = ep_data_in[dict(events=days_train_inds,channels=ep_data_in.channels[-1],
                                       time=0)].to_array().values.squeeze()
        sbj_order += [j]*dat_train.shape[0]
        
        if test_day is not None:
            dat_test = ep_data_in[dict(events=days_test_inds,channels=slice(0,n_chans_curr),
                                       time=time_inds)].to_array().values.squeeze()
            labels_test = ep_data_in[dict(events=days_test_inds,channels=ep_data_in.channels[-1],
                                          time=0)].to_array().values.squeeze()
            sbj_order_test += [j]*dat_test.shape[0]
            
        #Pad data in electrode dimension if necessary
        if (len(pats_ids_in) > 1) and (n_chans_all > n_ecog_chans):
            dat_sh = list(dat_train.shape)
            dat_sh[1] = n_chans_all
            #Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
            X_pad = np.zeros(dat_sh)
            X_pad[:,:n_ecog_chans,...] = dat_train
            dat_train = X_pad.copy()
            
            if test_day is not None:
                dat_sh = list(dat_test.shape)
                dat_sh[1] = n_chans_all
                #Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
                X_pad = np.zeros(dat_sh)
                X_pad[:,:n_ecog_chans,...] = dat_test
                dat_test = X_pad.copy()
        
        #Concatenate across subjects
        if j==0:
            X_subj = dat_train.copy()
            y_subj = labels_train.copy()
            if test_day is not None:
                X_test_subj = dat_test.copy()
                y_test_subj = labels_test.copy()
        else:
            X_subj = np.concatenate((X_subj,dat_train.copy()),axis=0)
            y_subj = np.concatenate((y_subj,labels_train.copy()),axis=0)
            if test_day is not None:
                X_test_subj = np.concatenate((X_test_subj,dat_test.copy()),axis=0)
                y_test_subj = np.concatenate((y_test_subj,labels_test.copy()),axis=0)
    
    sbj_order = np.asarray(sbj_order)
    sbj_order_test = np.asarray(sbj_order_test)
    print('Data loaded!')
    
    return X_subj,y_subj,X_test_subj,y_test_subj,sbj_order,sbj_order_test


def randomize_data(sp, X, y, sbj_order, overwrite=False):
    '''
    Randomize event order, with saving option for consistency
    '''
    savename = sp+'order_inds/gen_allpats_order_inds.npy'
    if path.exists(savename) and (not overwrite):
        #Reuse old order ind files, if exist
        order_inds = np.load(savename)
    else:
        order_inds = np.arange(len(y))
        np.random.shuffle(order_inds)
        np.save(savename,order_inds)
    X = X[order_inds,...]
    y = y[order_inds]
    sbj_order = np.asarray(sbj_order)
    sbj_order = sbj_order[order_inds]
    print('Randomized event order!')
    
    return X, y, sbj_order


def folds_choose_subjects(n_folds, sbj_ids_all, n_test=1, n_val=4, n_train=7):
    '''
    Randomly pick train/val/test subjects for each fold
    (Updated to assign test subject evenly across subjects (if n_test=1.)
    '''
    n_subjs = len(sbj_ids_all)
    sbj_inds_all_train,sbj_inds_all_val,sbj_inds_all_test = [],[],[]
    for frodo in range(n_folds):
        if n_test==1:
            #Assign test subject as evenly as possible (still done randomly, using random permutation)
            if frodo%n_subjs == 0:
                #New random permutation of test subjects after iterate through all subjects
                test_sbj_count = 0
                test_sbj_all = np.random.permutation(n_subjs)
                if n_val==1:
                    #Assign validation subject evenly too
                    val_sbj_all = np.zeros([n_subjs,])
                    while np.any(val_sbj_all==test_sbj_all):
                        #Generate permutation that doesn't overlap with test subjects
                        val_sbj_all = np.random.permutation(n_subjs)
            sbj_inds = np.arange(n_subjs)
            curr_test_ind = test_sbj_all[test_sbj_count]
            sbj_inds_all_test.append(np.asarray([sbj_inds[curr_test_ind]]))
            if n_val==1:
                curr_val_ind = val_sbj_all[test_sbj_count]
                sbj_inds_all_val.append(np.asarray([sbj_inds[curr_val_ind]]))
                sbj_inds = np.delete(sbj_inds,np.array([curr_test_ind,curr_val_ind]))
                np.random.shuffle(sbj_inds)
                sbj_inds_all_train.append(sbj_inds[:][:n_train])
            else:
                sbj_inds = np.delete(sbj_inds,curr_test_ind)
                np.random.shuffle(sbj_inds)
                sbj_inds_all_val.append(sbj_inds[:n_val])
                sbj_inds_all_train.append(sbj_inds[n_val:][:n_train])
            test_sbj_count += 1
        else:
            sbj_inds = np.arange(n_subjs)
            np.random.shuffle(sbj_inds)
            sbj_inds_all_test.append(sbj_inds[:n_test])
            sbj_inds_all_val.append(sbj_inds[n_test:(n_val+n_test)])
            sbj_inds_all_train.append(sbj_inds[(n_val+n_test):][:n_train])

    return sbj_inds_all_train,sbj_inds_all_val,sbj_inds_all_test

def subject_data_inds(sbj, sbj_order, labels_unique, frodo, save_string, half_n_evs, y, sp, n_folds, inds, overwrite=False):
    '''
    Determine the indices for train, val, or test sets, ensuring that:
        number of move events = number of rest events = half_n_evs
    '''
    for j in sbj.tolist():
        inds_tmp_orig = np.nonzero(sbj_order==j)[0] #select inds for 1 subject at a time
        y_labs = y[inds_tmp_orig]
        for i,lab_uni in enumerate(labels_unique):
            inds_tmp = inds_tmp_orig[y_labs==lab_uni]
            #Make randomization for each fold the same across models
            order_inds = np.arange(len(inds_tmp))
            np.random.shuffle(order_inds) #randomize order

            inds_tmp = inds_tmp[order_inds]
            if half_n_evs != 'nopad':
                if len(inds_tmp)<half_n_evs:
                    #Make length >= half_n_evs
                    inds_tmp = list(inds_tmp)*((half_n_evs//len(inds_tmp))+1)
                    inds_tmp = np.asarray(inds_tmp)[:half_n_evs]
                else:
                    inds_tmp = inds_tmp[:half_n_evs]
            
            if i==0:
                inds_sbj = inds_tmp.copy()
            else:
                inds_sbj = np.concatenate((inds_sbj,inds_tmp),axis=0)
                
        inds += list(inds_sbj)
    
    return np.asarray(inds)


def roi_proj_rf(X_in,sbj_order,nROIs,proj_mat_out):
    '''
    Project spectral power from electrodes to ROI's prior for random forest classification
    '''
    #Project to ROIs using matrix multiply
    X_in_sh = list(X_in.shape)
    X_in_sh[1] = nROIs
    X_in_proj = np.zeros(X_in_sh)
    for s in range(X_in.shape[0]):
        X_in_proj[s,...] = proj_mat_out[sbj_order[s],...] @ X_in[s,...]
    del X_in
    X_in_proj = X_in_proj.reshape(X_in_proj.shape[0],-1)
    
    return X_in_proj

def get_custom_motor_rois(regions=['precentral','postcentral','parietal_inf']):
    '''
    Returns ROI indices for those within the precentral, postcentral, and inferior parietal regions (accoring to AAL2)
    '''
    precentral_inds = [2263,2557,2558,2571,2587,2845,2846,2847,2858,2859,2873,2874,3113,3123,3124,3136,3137,3138,3151,3153,3154,3359,3360,3369,3370,3371,3383,3384,3559,3565,3566,3567,3568,3576,3577,3578,3579,3589,3590,3722,3723,3724,3729,3730,3731,3739,3740,3752,3837]
    postcentral_inds = [2236,2237,2238,2246,2247,2248,2545,2546,2547,2555,2556,2569,2570,2835,2836,2843,2844,2856,2857,2871,3110,3111,3112,3121,3122,3133,3134,3135,3149,3350,3351,3355,3356,3357,3358,3367,3368,3381,3382,3395,3555,3556,3557,3563,3564,3574,3575,3587,3588,3720,3721,3727,3728,3737,3738,3832,3834,3835,3836,3842,3843]
    parietal_inf_inds = [3106,3107,3108,3116,3117,3118,3119,3120,3131,3132,3143,3144,3145,3146,3147,3148,3161,3347,3348,3349,3352,3353,3354,3364,3365,3366,3376,3378,3379,3380,3553,3554,3561,3562]
    
    # Account for Matlab indexing starting at 1
    precentral_inds = [val-1 for val in precentral_inds]
    postcentral_inds = [val-1 for val in postcentral_inds]
    parietal_inf_inds = [val-1 for val in parietal_inf_inds]
    
#     custom_roi_inds = np.union1d(np.union1d(precentral_inds,postcentral_inds),parietal_inf_inds) #select for sensorimotor ROIs
    custom_roi_inds = []
    for val in regions:
        eval('custom_roi_inds.extend('+val+'_inds)')
    return custom_roi_inds

def str2bool(v):
    '''
    Allows True/False booleans in argparse
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def unseen_modality_test(eeg_lp, roi_proj_loadpath, ecog_root, pow_type='relative_power',
                         model_type = 'eegnet_hilb'):
    """
    Test trained same modality decoders on unseen EEG data.
    """
    model_lp = ecog_root + 'combined_sbjs_' + pow_type + '/'
    pats_ids_in = ['EE'+str(val).zfill(2) for val in np.arange(1, 16).tolist()]
    custom_rois = True
    n_chans_eeg = 61
    n_chans_ecog = 126  # number of channels in ecog data (expected by model)

    # Load param file from pre-trained model
    file_pkl = open(model_lp+'param_file.pkl', 'rb')
    params_dict = pickle.load(file_pkl)
    file_pkl.close()
    
    # Extract appropriate parameters from param file
    tlim = params_dict['tlim']
    test_day = params_dict['test_day']
    rand_seed = params_dict['rand_seed']
    n_test_sbj = params_dict['n_test']
    n_val_sbj = params_dict['n_val']
    n_folds = params_dict['n_folds']
    save_suffix = params_dict['save_suffix']
    do_log = params_dict['do_log']
    if 'n_train' in list(params_dict.keys()):
        n_train_sbj = params_dict['n_train']
    else:
        n_train_sbj = 7
    if 'epochs' in list(params_dict.keys()):
        epochs = params_dict['epochs']
        compute_val = params_dict['compute_val']
        ecog_srate = params_dict['ecog_srate']

    # Find pathnames of models from all folds
    if model_type in ['rf','riemann']:
        model_fnames = natsort.natsorted(glob.glob(model_lp + model_type+'_fold*.sav'))
    else:
        model_fnames = natsort.natsorted(glob.glob(model_lp + 'checkpoint_gen_'+model_type+'_fold*.h5'))

    # Load projection matrix
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
                                                                   roi_proj_loadpath=roi_proj_loadpath)
    nROIs = len(good_ROIs)
    print("ROIs found")

    accs = np.zeros([len(model_fnames),len(pats_ids_in)])
    for i,curr_pat in enumerate(pats_ids_in):
        # Load data
        X_all,y_all,_,_,sbj_order_all,_ = load_data([curr_pat], eeg_lp, test_day=None, tlim=tlim)
        X_all[np.isnan(X_all)] = 0 # set all NaN's to 0
        
        # Reformat data size for NN
        Y_test = np_utils.to_categorical(y_all-1)
        X_test_tmp = np.expand_dims(X_all,1)
        proj_mat_out2 = np.tile(proj_mat_out,[X_test_tmp.shape[0],1,1])
        proj_mat_out2 = np.expand_dims(proj_mat_out2,1)

        # Pad channel dimension to match ECoG data
        X_test_sh = list(X_test_tmp.shape)
        X_test_sh[2] = n_chans_ecog
        X_test = np.zeros(X_test_sh)
        X_test[...,:n_chans_eeg,:] = X_test_tmp
        proj_mat_out3 = np.zeros(list(proj_mat_out2.shape[:-1])+[n_chans_ecog])
        proj_mat_out3[...,:n_chans_eeg] = proj_mat_out2

        if model_type in ['rf','riemann']:
            X_test = X_test.squeeze()
            Y_test = y_all.copy()
            proj_mat_out3 = proj_mat_out3.squeeze()
            nROIs = proj_mat_out3.shape[1]
            
        for s in tqdm(range(len(model_fnames))):
            fID = model_fnames[s]

            if model_type=='rf':
                sbj_order_test = np.ones(X_test.shape[0]).astype('int')
                X_test_proj = roi_proj_rf(X_test, sbj_order_test, nROIs, proj_mat_out3)
                clf = pickle.load(open(fID, 'rb'))
                accs[s, i] = accuracy_score(Y_test.ravel(), clf.predict(X_test_proj))
            elif model_type == 'riemann':
                sbj_order_test = np.ones(X_test.shape[0]).astype('int')
                X_test_proj = roi_proj_rf(X_test, sbj_order_test, nROIs, proj_mat_out3)
                # Reshape into 3 dimensions
                X_test_proj2 = X_test_proj.reshape((X_test.shape[0],-1,X_test.shape[-1]))
                cov_data_test = pyriemann.estimation.Covariances('lwf').fit_transform(X_test_proj2)
                clf = pickle.load(open(fID, 'rb'))
                accs[s, i] = accuracy_score(Y_test.ravel(), clf.predict(cov_data_test))
            else:
                # Load pre-trained model
                pretrain_model = tf.keras.models.load_model(fID)
                
                # Test on EEG data
                preds = pretrain_model.predict([X_test, proj_mat_out3]).argmax(axis = -1) 
                accs[s, i] = np.mean(preds == Y_test.argmax(axis=-1))

    suffix = ''
    if model_type!='eegnet_hilb':
        suffix = '_'+model_type
    
    np.save(ecog_root+'accs_ecogtransfer_'+pow_type+suffix+'.npy',accs)
    
def roi_proj_pow(X_in,sbj_order,nROIs,proj_mat_out,ecog=True):
    """
    Project spectral power from electrodes to ROI's prior for random forest classification
    """
    #Project to ROIs using matrix multiply
    X_in_sh = list(X_in.shape)
    X_in_sh[1] = nROIs
    X_in_proj = np.zeros(X_in_sh)
    for s in range(X_in.shape[0]):
        sh_orig = X_in_proj.shape
        X_in_ep = X_in[s,...].reshape(X_in.shape[1],-1)
        if ecog:
            X_in_ep_proj = proj_mat_out[sbj_order[s],...] @ X_in_ep
        else:
            X_in_ep_proj = proj_mat_out[0,...] @ X_in_ep # EEG data has same projection matrix
        X_in_proj[s,...] = X_in_ep_proj.reshape(sh_orig[1:])
    del X_in,X_in_ep_proj,X_in_ep
    
    return X_in_proj

def compute_tfr(epochsAllMove,eventType,epoch_times,freqs = np.arange(6, 123, 3),crop_val=0.5,decim=30):
    """
    Computes spectrogram using Morlet wavelets (log-scaled).
    """
    n_cycles = freqs / 4.  # different number of cycle per frequency

    #Compute power for move trials
    print('Computing power...')
    power = tfr_morlet(epochsAllMove[eventType], freqs=freqs, n_cycles=n_cycles, use_fft=False,
                       return_itc=False, decim=decim, n_jobs=1,average=False)
    print('Power computation complete!')
    power.crop(epoch_times[0]+crop_val, epoch_times[1]-crop_val) #trim epoch to avoid edge effects
    power.data = 10*np.log10(power.data+\
                             np.finfo(np.float32).eps) #convert to log scale
    power.data[np.isinf(power.data)]=0 #set infinite values to 0
    return power

def diff_specs(lp, data_lp, ecog = True, roi_of_interest = 47, pad_val = 0.5,
               ecog_srate = 250, model_type = 'eegnet_hilb', decim_spec = 50):
    """
    Computes difference spectrograms for ECoG data.
    Inputs:
            ecog : plotting ECoG (True) or EEG (False) spectrograms
            roi_of_interest : index of ROI that we want to analyze difference spectrograms for
            pad_val : used for removing edge effects in spectrogram calculation (sec)
            ecog_srate : sampling rate of ECoG data (Hz)
    """
    nROIs = 1
    # Load model params file
    file_pkl = open(lp+'param_file.pkl', 'rb')
    params_dict = pickle.load(file_pkl)
    file_pkl.close()

    # Extract appropriate parameters from param file
    tlim = params_dict['tlim']
    tlim_orig = tlim.copy()
    tlim[0] -= pad_val
    tlim[1] += pad_val
    test_day = params_dict['test_day']
    if ecog:
        pats_ids_in = params_dict['pats_ids_in']
    else:
        pats_ids_in = ['EE'+str(val).zfill(2) for val in np.arange(1,16).tolist()]
    rand_seed = params_dict['rand_seed']
    n_test_sbj = params_dict['n_test']
    n_val_sbj = params_dict['n_val']
    model_lp = params_dict['sp']
    n_folds = params_dict['n_folds']
    n_evs_per_sbj = params_dict['n_evs_per_sbj']
    n_filts = params_dict['F1']
    kernLength = params_dict['kernLength']
    if 'n_train' in list(params_dict.keys()):
        n_train_sbj = params_dict['n_train']
    else:
        n_train_sbj = 7

    # Load projection matrix
    if ecog:
        proj_mat_out = np.load(lp+'proj_mat_out.npy')
        proj_mat_out = proj_mat_out[:,roi_of_interest:(roi_of_interest+1),:]
        n_chans_all = len(np.nonzero(proj_mat_out.reshape(-1,proj_mat_out.shape[-1]).mean(axis=0))[0])
    else:
        custom_rois = True
        n_chans_eeg = 61
        n_chans_ecog = 126 # number of channels in ecog data (expected by model)
        per_test_trials = 0.2 # percentage of EEG data to use for test set
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
                                                                       roi_proj_loadpath= data_lp+'proj_mat/')
        print("ROIs found")
        n_chans_all = n_chans_eeg
        proj_mat_out = proj_mat_out[:,roi_of_interest:(roi_of_interest+1),:]

    # Set random seed
    np.random.seed(rand_seed)

    # Load ECoG data for all subjects
    if ecog:
        X,y,X_test_orig,y_test_orig,sbj_order,sbj_order_test_load = load_data(pats_ids_in, data_lp,
                                                                              n_chans_all=n_chans_all,
                                                                              test_day=test_day, tlim=tlim)
        del X_test_orig # only interested in train data
    else:
        X,y,_,_,sbj_order,_ = load_data(pats_ids_in, data_lp,
                                        n_chans_all=n_chans_all,
                                        test_day=None, tlim=tlim)
        X[np.isnan(X)] = 0 # set all NaN's to 0

    # Identify the number of unique labels (or classes) present
    labels_unique = np.unique(y)
    nb_classes = len(labels_unique)
    if isinstance(n_evs_per_sbj,str):
        half_n_evs = n_evs_per_sbj
    else:
        half_n_evs = n_evs_per_sbj//len(labels_unique)
    half_n_evs_test = 'nopad' #avoids duplicating events (will take all available events)
    
    if ecog:
        # Choose subjects for training/validation/testing for every fold (random seed keeps this consistent to pre-trained data)
        sbj_inds_all_train, sbj_inds_all_val, sbj_inds_all_test = folds_choose_subjects(n_folds, pats_ids_in,
                                                                                        n_test=n_test_sbj, n_val=n_val_sbj,
                                                                                        n_train=n_train_sbj)

    ave_pow_diffs = []
    # Determine subjects in train/val/test sets for current fold
    n_subjs = len(pats_ids_in)
    train_sbj = np.arange(n_subjs)
    train_inds = []
    if ecog:
        train_inds = subject_data_inds(train_sbj, sbj_order, labels_unique, 0, 
                                       'train_inds', half_n_evs, y, '', n_folds, train_inds, True)

    # Generate train data based on event indices for each fold
    if ecog:
        X_train = X[train_inds,...] # shape (n_epochs, n_channels, n_times)
        Y_train = y[train_inds]
        sbj_order_train = sbj_order[train_inds] # important for projection matrix input
    else:
        X_train = X.copy() # shape (n_epochs, n_channels, n_times)
        Y_train = y.copy()
        sbj_order_train = sbj_order.copy() # important for projection matrix input


    n_filts = 1
    power_proj_diff = [[[] for j in range(n_filts)] for k in range(n_subjs)]
    for k,curr_train_sbj in enumerate(train_sbj):
        curr_ev_inds = np.nonzero(sbj_order_train==curr_train_sbj)[0]
        X_train_sbj = X_train[curr_ev_inds,...]
        Y_train_sbj = Y_train[curr_ev_inds]
        sbj_order_train_sbj = sbj_order_train[curr_ev_inds]

        # Create info for epochs array
        ch_names = ['ECoG-'+str(ind) for ind in range(X_train_sbj.shape[1])]
        ch_types = ['eeg']*X_train_sbj.shape[1]
        info = mne.create_info(ch_names=ch_names, sfreq=ecog_srate, ch_types=ch_types)

        # Filter data using Conv1D
        X_train_sbj = np.expand_dims(X_train_sbj,1)

        # Create epoched data prior to time-frequency computation
        events = np.zeros([len(Y_train_sbj),3])
        events[:,0] = np.arange(events.shape[0])
        events[:,2] = Y_train_sbj
        events = events.astype('int')
        event_dict = {'rest':1,'move':2}

        for j in range(n_filts):
            epochs = mne.EpochsArray(X_train_sbj[:,j,...], info=info, events=events, event_id=event_dict, tmin=tlim[0])

            # Compute and project power for move events
            power = compute_tfr(epochs,'move',tlim,freqs = np.arange(1, 124, 5),crop_val=pad_val,decim=decim_spec)
            power_move_proj = np.median(roi_proj_pow(power.data,sbj_order_train_sbj,nROIs,proj_mat_out,ecog),axis=0).squeeze()

            # Compute and project power for rest events
            power = compute_tfr(epochs,'rest',tlim,freqs = np.arange(1, 124, 5),crop_val=pad_val,decim=decim_spec)
            power_rest_proj = np.median(roi_proj_pow(power.data,sbj_order_train_sbj,nROIs,proj_mat_out,ecog),axis=0).squeeze()

            # Take difference of rest and move average power
            power_proj_diff[k][j] = power_move_proj - power_rest_proj

    # Reshape and take average across subjects and filters
    pow_sh = power_proj_diff[0][0].shape
    final_spec = np.asarray(power_proj_diff).reshape((-1,pow_sh[0],pow_sh[1])).mean(axis=0)

    # Create dummy power variable
    power.drop_channels(power.info['ch_names'][1:])
    power = power.average()

    # Take average power across folds
    # final_spec = np.asarray(ave_pow_diffs).mean(axis=0)
    if ecog:
        savename = 'diff_spec_'+model_type+'_tfr.h5'
    else:
        savename = 'diff_spec_'+model_type+'_eeg_tfr.h5'
    power.data[-2:] = final_spec # put data into dummy power variable

    # Save final spectrogram and time/frequencies
    power.save(lp+savename,overwrite=True)
    
def ntrain_combine_df(root_path, ntra = [1,10],
                      suffix_lp = '/combined_sbjs_ntra', acc_type = 'Test'):
    """
    Combines accuracies from training multiple participants into 1 dataframe
    Inputs:
            root_path : top-level directory of saved files
            ntra : min, max number of training subjects used
    """

    ntra_lst = np.arange(ntra[0],ntra[1]+1).tolist()
    lp = [root_path+suffix_lp+str(val)+'/' for val in ntra_lst]
    
    # Load parameters from param file
    file_pkl = open(lp[0]+'param_file.pkl', 'rb')
    params_dict = pickle.load(file_pkl)
    file_pkl.close()

    rand_seed = params_dict['rand_seed']
    n_folds = params_dict['n_folds']
    pats_ids_in = params_dict['pats_ids_in']
    combined_sbjs = params_dict['combined_sbjs']
    test_day = params_dict['test_day']
    models_used = params_dict['models']
    n_test = params_dict['n_test']
    n_val = params_dict['n_val']

    model_dict = {'eegnet_hilb':'HTNet','eegnet':'EEGNet','rf':'Random Forest',
                  'riemann':'Minimum Distance'} # Dictionary for plot legend
    
    # Determine train/val/test splits
    np.random.seed(rand_seed)
    sbj_inds_all_train, sbj_inds_all_val, sbj_inds_all_test = folds_choose_subjects(n_folds, pats_ids_in,
                                                                                    n_test=n_test, n_val=n_val)
    sbj_inds_all_test_sm = [val[0] for val in sbj_inds_all_test]
    test_sbj_folds = np.asarray(sbj_inds_all_test_sm)
    
    # Load in accuracy values across folds
    acc_types = ['Train','Val','Test']
    acc_ind = np.nonzero(np.asarray(acc_types)==acc_type)[0]
    n_sbj_amts = len(lp)
    n_models = len(models_used)
    n_subjs = len(pats_ids_in)
    accs_all = np.zeros([n_folds,n_sbj_amts,n_models]) #middle value is train,val,test accuracies
    accs_all[:] = np.nan
    for i,model_type in enumerate(models_used):
        for j in range(n_sbj_amts):
            tmp_vals = np.load(lp[j]+'acc_gen_'+model_type+'_'+str(n_folds)+'.npy')
            for p in range(n_folds):
                accs_all[p,j,i] = tmp_vals[p,acc_ind]
                
    # Average results for each participant
    ave_vals_test_sbj = np.zeros([n_subjs,n_sbj_amts,n_models])
    for sbj in range(n_subjs):
        folds_sbj = np.nonzero(test_sbj_folds==sbj)[0]
        for j,modtype in enumerate(models_used): 
            ave_vals_test_sbj[sbj,:,j] = np.mean(accs_all[folds_sbj,:,j],axis=0)

    # Reshape to 2D array for pandas dataframe
    dat_sh = ave_vals_test_sbj.shape
    ave_vals_2d = np.zeros([dat_sh[0],dat_sh[1]*dat_sh[2]])
    for i in range(dat_sh[2]):
        ave_vals_2d[:,(dat_sh[1]*i):(dat_sh[1]*(i+1))] = ave_vals_test_sbj[:,:,i]

    patIDs_sm = [val[:3] for val in pats_ids_in]
    patIDs_sm_cons = []
    for val in patIDs_sm:
        patIDs_sm_cons.extend([val]*n_sbj_amts*n_models)

    mod_ids = []
    for i in range(n_subjs):
        for mod_curr in models_used:
            mod_ids.extend([mod_curr]*n_sbj_amts)
    mod_ids = [model_dict[val] for val in mod_ids]

    n_tra_lst = [str(val) for val in ntra_lst]*n_models*n_subjs


    vals_np = np.asarray([ave_vals_2d.flatten().tolist()]+[patIDs_sm_cons]+\
                         [mod_ids]+[n_tra_lst]).T
    df_sbj = pd.DataFrame(vals_np, columns=['Accuracy','sbj','Models','n_tra'])
    df_sbj['Accuracy'] = pd.to_numeric(df_sbj['Accuracy'])
    df_sbj['n_tra'] = pd.to_numeric(df_sbj['n_tra'])

    df_sbj.to_csv(root_path+'/ntra_df.csv')

def proj_compute_dipdens(patient_ids, roi_proj_loadpath,
                         atlas = 'none'):
    """
    Loads projection matrix for each subject and extracts dipole densities (top row)
    
    Inputs:
            patient_ids : which participants to get projection matrix from
            roi_proj_loadpath : where to load projection matrix CSV files
            atlas : ROI projection atlas to use (aal, loni, brodmann, or none)
    """
    #Find good ROIs first
    dipole_densities = []
    for s,patient in enumerate(patient_ids):
        df = pd.read_csv(roi_proj_loadpath+atlas+'_'+patient+'_elecs2ROI.csv')
        dip_dens = df.iloc[0]
        dipole_densities.append(dip_dens)

    return np.asarray(dipole_densities)
    
def frac_combine_df(root_path, roi_proj_loadpath, dipole_dens_thresh = .07, accuracy_to_plot = 'Test',
                    custom_rois = True, compare_frac = 'train_test',
                    custom_rois_compare = ['precentral','postcentral','parietal_inf'], meas = 'power'):
    """
    Combines accuracies and fraction overlap into 1 dataframe
    Inputs:
            dipole_dens_thresh : value to threshold dipole density values (higher means fewer ROI's get through)
            accuracy_to_plot : which accuracy values to plot on y axis ('Train','Val',or 'Test')
            custom_rois : if used custom ROI's during classification, limit the ROI's here to just those custom ones
            compare_frac : which participants to compare for fraction overlap
    """
    lp = root_path+'/combined_sbjs_'+meas
    # Load parameters from param file
    file_pkl = open(lp+'/param_file.pkl', 'rb')
    params_dict = pickle.load(file_pkl)
    file_pkl.close()
    rand_seed = params_dict['rand_seed']
    n_folds = params_dict['n_folds']
    pats_ids_in = params_dict['pats_ids_in']
    models_used = params_dict['models']
    n_test = params_dict['n_test']
    n_val = params_dict['n_val']
    
    # Load model accuracies
    acc_types = ['Train','Val','Test']
    n_accs = len(acc_types)
    n_models = len(models_used)
    accs_all = np.zeros([n_folds,n_accs,n_models]) #middle value is train,val,test accuracies
    accs_all[:] = np.nan
    for i,model_type in enumerate(models_used):
        tmp_vals = np.load(lp+'/acc_gen_'+model_type+'_'+str(n_folds)+'.npy')
        for j in range(n_accs):
            for p in range(n_folds):
                accs_all[p,j,i] = tmp_vals[p,j]
    
    if custom_rois:
        custom_roi_inds = get_custom_motor_rois()
            
    for mod_ind in range(len(models_used)):
        #Load dipole densities for all subjects
        dipole_dens = proj_compute_dipdens(pats_ids_in, roi_proj_loadpath)
        
        # Determine train/val/test splits
        np.random.seed(rand_seed)
        sbj_inds_all_train, sbj_inds_all_val, sbj_inds_all_test = folds_choose_subjects(n_folds, pats_ids_in,
                                                                                        n_test=n_test, n_val=n_val)
        sbj_inds_all_train_np = np.asarray(sbj_inds_all_train)
        sbj_inds_all_val_np = np.asarray(sbj_inds_all_val)
        sbj_inds_all_test_np = np.asarray(sbj_inds_all_test)

        #Determine which subjects to compare for every fold, based on compare_frac specification
        sbjs_numerator = sbj_inds_all_test_np
        if compare_frac == 'train_test':
            sbjs_denominator = sbj_inds_all_train_np
        elif compare_frac == 'trainval_test':
            sbjs_denominator = np.concatenate((sbj_inds_all_train_np,sbj_inds_all_val_np),axis=1)
        elif compare_frac == 'motor_area':
            sbjs_denominator = sbj_inds_all_train_np

        frac_overlap = []
        for i in range(n_folds):        
            mean_dips1 = dipole_dens[sbjs_numerator[i,:],:].mean(axis=0)
            inds_thresh1 = np.nonzero(mean_dips1 >= dipole_dens_thresh)[0]
            if compare_frac == 'motor_area':
                custom_roi_inds_compare = get_custom_motor_rois(custom_rois_compare)
                inds_thresh2 = custom_roi_inds_compare.copy()
            else:
                mean_dips2 = dipole_dens[sbjs_denominator[i,:],:].mean(axis=0)
                inds_thresh2 = np.nonzero(mean_dips2 >= dipole_dens_thresh)[0]

            if custom_rois:
                frac_num = len(reduce(np.intersect1d, (inds_thresh1,inds_thresh2,custom_roi_inds)))
                frac_denom = len(reduce(np.intersect1d, (inds_thresh2,custom_roi_inds)))
            else:
                frac_num = len(reduce(np.intersect1d, (inds_thresh1,inds_thresh2)))
                frac_denom = len(inds_thresh2)
            
            frac_overlap.append(frac_num/frac_denom)

        # Add results to dataframe
        frac_overlap_np = np.asarray(frac_overlap)
        acc_plt_dict = {'Train':0,'Val':1,'Test':2}
        acc_plt = accs_all[:,acc_plt_dict[accuracy_to_plot],mod_ind]
        
        model_dict = {'eegnet_hilb':'HTNet','eegnet':'EEGNet','rf':'Random Forest',
                      'riemann':'Minimum Distance'}
        col_labels = [model_dict[val] for val in models_used]
        mod_df_lst = [[col_labels[mod_ind]]*len(acc_plt)]
        if mod_ind==0:
            df_plt = pd.DataFrame([frac_overlap_np,acc_plt]+mod_df_lst).T
            df_plt.columns=['Frac','Acc','Model']
        else:
            df_tmp = pd.DataFrame([frac_overlap_np,acc_plt]+mod_df_lst).T
            df_tmp.columns=['Frac','Acc','Model']
            df_plt = pd.concat([df_plt,df_tmp],ignore_index=True)
            
    df_plt['Acc'] = pd.to_numeric(df_plt['Acc'])
    df_plt['Frac'] = pd.to_numeric(df_plt['Frac'])
    
    df_plt.to_csv(root_path+'/frac_overlap_df.csv')
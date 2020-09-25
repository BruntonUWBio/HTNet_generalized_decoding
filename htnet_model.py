"""
Contains the keras model for HTNet and EEGNet decoding.
EEGNet code written by @vlawhern (https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py).
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten, Concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

# Load utility functions for custom HTNet layers
from model_utils import apply_hilbert_tf, proj_to_roi

def htnet(nb_classes, Chans = 64, Samples = 128, 
          dropoutRate = 0.5, kernLength = 64, F1 = 8, 
          D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout',
          ROIs = 100,useHilbert=False,projectROIs=False,kernLength_sep = 16,
          do_log=False,compute_val='power',data_srate = 500,base_split = 4):
    """
    Keras model for HTNet, which implements EEGNet with custom layers that implement
    the hilbert transform to compute spectral power/phase/frequency and project
    data into common brain regions to generalize across participants, even when
    electrode placement varies widely.
    
    Inputs:
        
      nb_classes      : Number of classes to classify
      Chans, Samples  : Number of channels and time points in the input neural data
      dropoutRate     : Dropout fraction
      kernLength      : Length of temporal convolution kernel in first layer
      F1, F2          : Number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn; we used F2 = F1 * D, same as EEGNet paper 
      D               : Number of spatial filters to learn within each temporal
                        convolution
      norm_rate       : Maximum norm for dense layer weights
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string
      ROIs            : Number common brain regions projecting to (only used if projectROIs == True)
      useHilbert      : If true, use Hilbert transform layer (HTNet); if false, will 
                        decode using time-domain signal (EEGNet)
      projectROIs     : If true, project electrode-level data to common brain regions of interest,
                        using projection matrix (2nd input when fitting model)
      kernLength_sep  : Length of temporal convolution kernel in separable convolution layer
      do_log          : If true, will compute log(x+1) of spectral power (only used if useHilbert ==True
                        and compute_val == 'power')
      compute_val     : Spectral measure to compute (if useHilbert ==True); can be 'power', 'relative_power', 'phase', or 'freqslide'
                        for instantaneous power, relative power, phase, or frequency, respectively
      data_srate      : Sampling rate of neural data for instantaneous frequency computation (if useHilbert ==True
                        and compute_val == 'freqslide')
      base_split      : Determines baseline to use for relative power; averages time dimension based on base split
                        and takes first segment as baseline
    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (1, Chans, Samples))
    if projectROIs:
        input2   = Input(shape = (1, ROIs, Chans))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (1, Chans, Samples),
                                   use_bias = False)(input1)
    
    if useHilbert:
        # Hilbert transform
        if compute_val == 'relative_power':
            # Compute power for filtered input and divide by power for raw input
            X1 = Lambda(apply_hilbert_tf, arguments={'do_log':True,'compute_val':'power'})(block1) 
            
            # Subtract off baseline (at beginning of input data trials)
            X2 = AveragePooling2D((1, X1.shape[-1]//base_split))(X1) # average across all time points
            X2 = Lambda(lambda x: tf.tile(x[...,:1],tf.constant([1,1,1,Samples], dtype=tf.int32)))(X2)
            block1 = Lambda(lambda inputs: inputs[0]-inputs[1])([X1, X2])
        else:
            block1       = Lambda(apply_hilbert_tf, arguments={'do_log':do_log,'compute_val':compute_val,\
                                                               'data_srate':data_srate})(block1)
    if projectROIs:
        # Project to common brain regions
        # block1       = AveragePooling2D((1, 2))(block1) # can downsample spectral measure before projection step to limit memory usage
        block1       = Lambda(proj_to_roi)([block1,input2]) #project to ROIs
    block1       = BatchNormalization(axis = 1)(block1)
    
    # Depthwise kernel acts over all electrodes or brain regions
    if projectROIs:
        block1       = DepthwiseConv2D((ROIs, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    else:
        block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, kernLength_sep),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    if projectROIs:
        # Projecting to common brain regions requires weight matrix input
        return Model(inputs=[input1,input2], outputs=softmax)
    else:
        return Model(inputs=input1, outputs=softmax)
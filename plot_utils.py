"""
Utility functions for all Jupyter notebook plots.
"""
import joblib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.lines as mlines
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
import glob,pdb,sys,os,pickle,copy,natsort
from matplotlib import rcParams
rcParams['font.family'] = 'arial'

from functools import reduce
import pingouin as pg
from nilearn import plotting as ni_plt
from scipy.signal import welch
from scipy.fftpack import fft
from scipy.signal import hann
from tqdm.notebook import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #Use GPU
import tensorflow as tf
from tensorflow.keras.models import Model
from keras import backend as K
from tensorflow.keras.layers import Input,Conv2D
from mne import read_epochs
from mne.time_frequency import read_tfrs

# Custom imports
from htnet_model import htnet
from model_utils import folds_choose_subjects,get_custom_motor_rois,proj_mats_good_rois

def plot_model_accs(root_path, dataset, nb_classes=2, compare_models=True,
                    dpi_plt=300):
    """
    Plot test accuracy across decoder types.
    Inputs:
        nb_classes : number of label types (currently set to 2: move, rest)
        compare_models : if True, plot across different models; if False, plot across different HilbertNet measures
    """
    
#     root_path = '/data2/users/stepeter/cnn_hilb_datasets/'
#     dataset = 'naturalistic_v3'
    replace_str = '___'
    suffix_lp = '/'+replace_str+'/' # single_sbjs, combined_sbjs

    test_types = ['single_sbjs','combined_sbjs','ecog2eeg']
    suffix_lst = ['Power','Log\nPower','Relative\nPower','Phase','Frequency'] #Needed if models used are the same (for axis label)
    n_eeg_sbjs = 15
    conds = ['Tailored decoder','Generalized decoder,\nsame modality',
             'Generalized decoder,\nunseen modality']
    cond_lets = ['(A)','(B)\n','(C)\n']
    conds2 = ['(D)','(E)','(F)']
    
    # Load variables from param file
    file_pkl = open(root_path+dataset+'/combined_sbjs_power/param_file.pkl', 'rb')
    params_dict = pickle.load(file_pkl)
    file_pkl.close()

    rand_seed = params_dict['rand_seed']
    n_folds = params_dict['n_folds']
    pats_ids_in = params_dict['pats_ids_in']
    combined_sbjs = params_dict['combined_sbjs']
    test_day = params_dict['test_day']
    n_test = params_dict['n_test']
    n_val = params_dict['n_val']
    lp = [root_path+dataset+suffix_lp[:-1]+'_power/']

    if compare_models:
        models_used = params_dict['models']
        lp *= len(models_used)
    else:
        lp = [root_path+dataset+suffix_lp,root_path+dataset+suffix_lp[:-1]+'_power_log/',
              root_path+dataset+suffix_lp[:-1]+'_relative_power/',
              root_path+dataset+suffix_lp[:-1]+'_phase/',root_path+dataset+suffix_lp[:-1]+'_freqslide/']
        models_used = ['eegnet_hilb']*len(lp)

    model_dict = {'eegnet_hilb':'HTNet','eegnet':'EEGNet','rf':'Random\nForest',
                  'riemann':'Minimum\nDistance','phase':'Phase','relative_power':'Rel.\nPower',
                  'freqslide':'Freq.','power_log':'Log\nPower','':'Power'} # Dictionary for plot legend
    
    # Determine test fold subject(s) using saved parameters
    np.random.seed(rand_seed)
    sbj_inds_all_train, sbj_inds_all_val, sbj_inds_all_test = folds_choose_subjects(n_folds, pats_ids_in,
                                                                                    n_test=n_test, n_val=n_val)
    sbj_inds_all_test_sm = [val[0] for val in sbj_inds_all_test]
    test_sbj_folds = np.asarray(sbj_inds_all_test_sm)
    
    # Create labels for ECoG and EEG participants
    ecog_sbjs,eeg_sbjs = [],[]
    for i in range(len(pats_ids_in)):
        ecog_sbjs.append('EC'+str(i+1).zfill(2))
    for i in range(n_eeg_sbjs):
        eeg_sbjs.append('EE'+str(i+1).zfill(2))

    # Create labels for model types or measures computed
    col_labels = [model_dict[val] for val in models_used]
    if len(np.unique(np.asarray(col_labels))) != len(col_labels):
        #If there are duplicate column labels, add different suffix strings
        col_labels = [suffix_lst[i] for i,val in enumerate(col_labels)]
        
    # Plot data
    title_pads = [18,8,8]
    title_y = [1.2,1.125,1.125]
    hspace_val = .8 if compare_models else .6
    use_asterisks = True
    fig,ax = plt.subplots(2,len(test_types),
                          dpi=dpi_plt,figsize=(7.5,4),
                          gridspec_kw={'wspace':.1, 'hspace':hspace_val})

    acc_types = ['Train','Val','Test']
    test_ind = np.nonzero(np.asarray(acc_types)=='Test')[0]
    dfs_all = []
    for ii,test_type in enumerate(test_types):
        n_accs = len(acc_types)
        n_models = len(models_used)
        if test_type!='single_sbjs':
            if test_type=='ecog2eeg':
                accs_all = np.zeros([n_eeg_sbjs,n_accs,n_models]) #middle dim is train,val,test accuracies
            else:
                accs_all = np.zeros([n_folds,n_accs,n_models])
            accs_all[:] = np.nan
            for i,model_type in enumerate(models_used):
                lp_curr = lp[i]
                if test_type=='ecog2eeg':
                    lp_curr = '/'.join(lp_curr.split('/')[:-2])+'/'
                    mod_type_curr = '' if model_type=='eegnet_hilb' else '_'+model_type
                    if not compare_models:
                        lp_spl = lp_curr.split('/')
                        lp_spl[:-2]+lp_spl[-1:]
                        lp_curr = '/'.join(lp_spl[:-2]+lp_spl[-1:])
                        suff = lp_spl[-2]
                    else:
                        suff = '_power'
                        ##########Use relative power instead for comparison##########
                        if mod_type_curr=='':
                            suff = '_relative_power'
                        ##########Use relative power instead for comparison##########
                    tmp_vals = np.load(lp_curr+'accs_ecogtransfer'+suff+mod_type_curr+'.npy')
                    for p in range(n_eeg_sbjs):
                        accs_all[p,test_ind,i] = np.mean(tmp_vals[:,p])
                else:
                    lp_curr = lp_curr.replace(replace_str,test_type)
                    tmp_vals = np.load(lp_curr+'acc_gen_'+model_type+'_'+str(n_folds)+'.npy')
                    for j in range(n_accs):
                        for p in range(n_folds):
                            accs_all[p,j,i] = tmp_vals[p,j]
        else:
            n_folds_sbj = n_folds//len(pats_ids_in)
            accs_all = np.zeros([len(pats_ids_in),n_accs,n_models,n_folds_sbj])
            accs_all[:] = np.nan
            for i,model_type in enumerate(models_used):
                for s,pat_curr in enumerate(pats_ids_in):
                    lp_curr = lp[i]
                    lp_curr = lp_curr.replace(replace_str,test_type)
                    tmp_vals = np.load(lp_curr+'acc_'+model_type+'_'+pat_curr+'_testday_'+str(test_day)+'.npy')
                    for j in range(n_accs):
                        for p in range(n_folds_sbj):
                            accs_all[s,j,i,p] = tmp_vals[p,j]

        # Average results for each participant
        if test_type=='ecog2eeg':
            ave_vals_test_sbj = accs_all[:,test_ind,:].squeeze()
        else:
            n_subjs = len(pats_ids_in)
            if test_type == 'combined_sbjs':
                n_folds_sbj = n_folds//n_subjs

            ave_vals_test = np.zeros([n_subjs*n_folds_sbj,len(models_used)])
            ave_vals_test_sbj = np.zeros([n_subjs,len(models_used)])

            if test_type == 'combined_sbjs':
                for sbj in range(len(pats_ids_in)):
                    folds_sbj = np.nonzero(test_sbj_folds==sbj)[0]
                    for j,modtype in enumerate(models_used): 
                        ave_vals_test_sbj[sbj,j] = round(np.mean(accs_all[folds_sbj,test_ind,j],axis=0),2)
                        for k in range(n_folds_sbj):
                            ave_vals_test[sbj+(k*n_subjs),j] = accs_all[folds_sbj[k],test_ind,j]   
            else:
                for i in range(accs_all.shape[0]):
                    for j in range(accs_all.shape[2]):
                        ave_vals_test_sbj[i,j] = round(np.mean(accs_all[i,test_ind,j,:],axis=-1)[0],2)
                        for k in range(n_folds_sbj):
                            ave_vals_test[i+(k*n_subjs),j] = accs_all[i,test_ind,j,k]

        ## Boxplots (top row)
        if test_type=='ecog2eeg':
            curr_subjs = eeg_sbjs
        else:
            curr_subjs = ecog_sbjs
        patIDs_lst = []
        for val in curr_subjs:
            patIDs_lst.extend([val]*len(models_used))

        if (test_type=='ecog2eeg') & compare_models:
            col_labels_plt = ['HilbertNet\n(Rel. Power)' if val=='HilbertNet' else val for val in col_labels]
        else:
            col_labels_plt = col_labels.copy()
        vals_np = np.asarray([ave_vals_test_sbj.flatten().tolist()]+[patIDs_lst]+\
                             [col_labels_plt*len(curr_subjs)]).T
        df_sbj = pd.DataFrame(vals_np, columns=['Accuracy','sbj','Models'])
        df_sbj['Accuracy'] = pd.to_numeric(df_sbj['Accuracy'])

        df = pd.DataFrame(ave_vals_test_sbj,index=curr_subjs,columns=col_labels_plt)
        ax[0,ii].axhline(1/nb_classes,c='k',linestyle='--')
        palette = sns.color_palette("colorblind",n_colors=df.shape[1])
        palette2 = palette.copy()
        palette2.reverse()
        sns.boxplot(data=df,ax=ax[0,ii],palette=palette,showfliers=False,whis=0)
    #     palette = sns.color_palette("Paired")
        sns.swarmplot(x='Models', y='Accuracy', s=4, data=df_sbj, ax=ax[0,ii],color='k') #, palette=palette, hue='sbj')
        # ax.legend_.remove()

        ax[0,ii].set_ylabel('Test Accuracy',fontsize=9)
        ax[0,ii].set_ylim([(1/nb_classes)-.05,1.065])
        ax[0,ii].set_yticks([.5,.75,1])
        ax[0,ii].spines['right'].set_visible(False)
        ax[0,ii].spines['top'].set_visible(False)
        ax[0,ii].spines['left'].set_bounds((1/nb_classes)-.05, 1)
        ax[0,ii].tick_params(axis='both', labelsize=8)
        ax[0,ii].set_xticklabels(ax[0,ii].get_xticklabels(), rotation = 40, ha="center")
        ax[0,ii].set_xlabel('')
        if ii>0:
            ax[0,ii].set_ylabel('')
            ax[0,ii].set_yticklabels([])
        ax[0,ii].tick_params(axis='x', pad=0)
        ax[0,ii].set_title(cond_lets[ii],fontsize=10,x=0,fontweight='bold',pad=title_pads[ii],color='dimgray')
        ax[0,ii].text((len(lp)/2)-.5,title_y[ii],conds[ii],fontsize=10,ha='center',fontweight='bold')
#         # Add stats (manually created)
#         star_fontsize = 18
#         if (ii==0) & compare_models:
#             y_start,h,w_x = 1.02,.04,1
#             ax[0,ii].plot([0, 0, 2, 2, 2-w_x, 2+w_x],
#                           [y_start, y_start+h, y_start+h, y_start, y_start, y_start],
#                           lw=1.5, c='k')
#             if use_asterisks:
#                 ax[0,ii].text(1,1.04,'*',fontsize=star_fontsize,fontweight='bold',ha='center')
#             else:
#                 ax[0,ii].text(1,1.09,r'$p<0.05$',fontsize=7,fontweight='normal',ha='center')
#         elif (ii==1) & compare_models:
#             y_start,h1,h2,w_x = .95,.06,.06,1
#             ax[0,ii].plot([0, 0, 2, 2, 2-w_x, 2+w_x],
#                           [y_start, y_start+h1, y_start+h1, y_start-h2, y_start-h2, y_start-h2],
#                           lw=1.5, c='k')
#             if use_asterisks:
#                 ax[0,ii].text(1,.99,'*',fontsize=star_fontsize,fontweight='bold',ha='center')
#             else:
#                 ax[0,ii].text(1,1.05,r'$p<0.05$',fontsize=7,fontweight='normal',ha='center')
#         elif (ii==2) & compare_models:
#             y_start,h1,h2,w_x = .8,.06,.1,1
#             ax[0,ii].plot([0, 0, 2, 2, 2-w_x, 2+w_x],
#                           [y_start, y_start+h1, y_start+h1, y_start-h2, y_start-h2, y_start-h2],
#                           lw=1.5, c='k')
#             if use_asterisks:
#                 ax[0,ii].text(1,.85,'***',fontsize=star_fontsize,fontweight='bold',ha='center')
#             else:
#                 ax[0,ii].text(1,.9,r'$p<0.001$',fontsize=7,fontweight='normal',ha='center')
#         if (ii==0) & (not compare_models):
#             ax[0,ii].text(2,.82,'**',fontsize=star_fontsize,fontweight='bold',ha='center')
#             ax[0,ii].text(3,.97,'**',fontsize=star_fontsize,fontweight='bold',ha='center')
#         elif (ii==1) & (not compare_models):
#             y_start,h = .92,.06
#             ax[0,ii].plot([2, 2, 4, 4],
#                           [y_start, y_start+h, y_start+h, y_start],
#                           lw=1.5, c='k')
#             ax[0,ii].text(3,.97,'**',fontsize=star_fontsize,fontweight='bold',ha='center')
#         elif (ii==2) & (not compare_models):
#             ax[0,ii].text(1,.55,'**',fontsize=star_fontsize,fontweight='bold',ha='center')
#             ax[0,ii].text(2,.8,'**',fontsize=star_fontsize,fontweight='bold',ha='center')
#             ax[0,ii].text(3,.65,'**',fontsize=star_fontsize,fontweight='bold',ha='center')

        ## Subject-by-subject lineplots (bottom row)
        col_labels_plt = df_sbj['Models'].unique().tolist().copy()
        if compare_models:
            col_labels_plt.reverse()
        ax[1,ii].axhline(1/nb_classes,c='k',linestyle='--')
        sns.lineplot(x='sbj',y='Accuracy',hue='Models',data=df_sbj,
                 ax=ax[1,ii],marker='o',markersize=5,linewidth=1,
                 hue_order=col_labels_plt,palette=palette2)
        leg = ax[1,ii].legend()
        leg_lines = leg.get_lines()
        for i in range(len(models_used)):
            ax[1,ii].lines[i+1].set_linestyle("None") # 'None') #
            leg_lines[i].set_linestyle("None") # 'None') #
        # plt.xticks(np.arange(1,13))

        ax[1,ii].set_ylim([(1/nb_classes)-.05,1])
        ax[1,ii].set_ylabel('Test Accuracy',fontsize=9)
        ax[1,ii].set_yticks([.5,.75,1])
        ax[1,ii].spines['right'].set_visible(False)
        ax[1,ii].spines['top'].set_visible(False)
        ax[1,ii].tick_params(axis='both', labelsize=8)
        ax[1,ii].legend_.remove()
        ax[1,ii].set_xlabel('')
        ax[1,ii].set_title(conds2[ii],fontsize=10,fontweight='bold',pad=8,color='dimgray', x=0)
        for tick in ax[1,ii].get_xticklabels():
                tick.set_rotation(60)
        ax[1,ii].tick_params(axis='x', pad=0)
        if ii>0:
            ax[1,ii].set_ylabel('')
            ax[1,ii].set_yticklabels([])

        # Save dataframe for stats
        dfs_all.append(df_sbj)
    plt.show()
    return fig, dfs_all
    
def test_acc_mod_stats(dfs_all):
    """
    Compute non-parametric stats for decoder accuracies.
    """
    for k,df_sbj in enumerate(dfs_all):
        print('')
        print(conds[k])
        # Friedman test (non-parametric, repeated measures ANOVA)
        print(pg.friedman(data=df_sbj, dv='Accuracy', within='Models', subject='sbj')['p-unc'])

        # Wilcoxon tests (non-parametric t-tests)
        p_vals = []
        col_labels = np.unique(df_sbj['Models'].values).tolist()
        n_models = len(col_labels)
        for i in range(n_models):
            for j in range(i+1,n_models):
                val1 = df_sbj[df_sbj['Models'] == col_labels[i]].iloc[:,0].values
                val2 = df_sbj[df_sbj['Models'] == col_labels[j]].iloc[:,0].values
                p_vals.append(float(pg.wilcoxon(val1,
                                                val2)['p-val']))

        # Correct for multiple comparisons
        _,p_vals = pg.multicomp(np.asarray(p_vals), alpha=0.05, method='fdr_bh')

        pval_df = np.zeros([n_models,n_models])
        q = 0
        for i in range(n_models):
            for j in range(i+1,n_models):
                pval_df[i,j] = p_vals[q]
                q += 1

        # Create output df with p_values
        df_pval = pd.DataFrame(pval_df,columns=col_labels,index=col_labels)
        df_pval[df_pval==0] = np.nan
        print(df_pval)

        
def plot_weights_interp(root_path, dataset, roi_proj_lp, spec_lp, mod_type = 'eegnet_hilb', nb_classes = 2,
                        dat_type = 'train', temp_conv_layer = 'conv2d', weight_type = 'mean',
                        srate = 250, bode_type = 'mag', custom_rois = True, dpi_plt = 300,
                        vscale_val = 1.5, epoch_times = [-1,1.001]):
    """
    HTNet decoder interpretability plots from analyzing convolutional layers.
    Inputs:
            spec_lp : path to pre-computed spectrogram data
            dat_type : which accuracies to use when weighting temporal frequency response ('train' or 'test')
            temp_conv_layer : which temporal convolutional layer to compute frequency response of ('conv2d' or 'separable_conv2d')
            weight_type : way to average temporal frequency response across filters ('max','mean','median','weighted','weighted_percentile');
                          only weighted and weighted_percentile use the accuracies from dat_type input
            srate : sampling rate of neural data used to train model (Hz)
            bode_type : look at magnitude ('mag') or phase ('phase') frequency response
            custom_rois : use predefined sensorimotor ROI's
    """
#     models_compare = ['eegnet_hilb']
#     root_path = '/data2/users/stepeter/cnn_hilb_datasets/'
#     dataset = 'naturalistic_v3'
#     roi_proj_lp = '/data1/users/stepeter/mvmt_init/ROIproj_matlab_smallROIs/'
    lp = root_path+dataset+'/combined_sbjs_power/'
    colors_plt = ['b','r']
    normalize_weighted_dat = False
    model_dict = {'eegnet_hilb':'Hilbert NN','eegnet':'Original NN','rf':'Random Forest'} # Dictionary for plot legend

    # Spatial filter plot parameters
    zero_sub_rand_accs = False # if True, set accuracies at or below random guessing to 0
    rand_acc_thresh = 0.5 # accuracy threshold for random guessing
    vmin,vmax = 0,1

    ## Temporal frequency response plot
    fig,axes=plt.subplots(2,2, figsize=(7.5,5), dpi=dpi_plt,
                          gridspec_kw={'wspace':.2, 'hspace':.47})
    pos = axes[0,1].get_position()
    width, height, x0, y0 = pos.width, pos.height, pos.x0, pos.y0
    axes[0,1].set_position([x0-.06,y0,width,height])
    print(temp_conv_layer+': '+weight_type)
    file_pkl = open(lp+'param_file.pkl', 'rb')
    params_dict = pickle.load(file_pkl)
    file_pkl.close()
    n_folds = params_dict['n_folds']
    save_suffix = params_dict['save_suffix']
    combined_sbjs  = params_dict['combined_sbjs']
    F1  = params_dict['F1']
    D  = params_dict['D']
    kernLength_sep  = params_dict['kernLength_sep']
    test_day = params_dict['test_day']
    pats_ids_in = params_dict['pats_ids_in']

    if not combined_sbjs:
        n_folds_sbj = n_folds
        n_folds *= len(pats_ids_in)

    weighted_pow_medians_all = []
    for curr_fold in tqdm(range(n_folds)):
        if weight_type == 'weighted':
            #Load CSV file with accuracies
            df = pd.read_csv(lp+'df_acc_allbut1_'+str(curr_fold)+'_'+mod_type+'_'+dat_type+'.csv')

            #Take off first filt ind, which is temporal filt number
            filts_left = []
            for i in range(len(df)):
                exec('tmp='+df['Filt_left'][i])
                filts_left.append(tmp[0])
            filts_left = np.asarray(filts_left)
            filts_left_r = []
            for i in range(len(df)):
                exec('tmp='+df['Filt_left'][i])
                filts_left_r.append(tmp[1])
            filts_left_r = np.asarray(filts_left_r)

        #Find important frequencies in temporal filters
        #Load model
        if combined_sbjs:
            loadname = lp+'checkpoint_gen_'+mod_type+'_fold'+str(curr_fold)+save_suffix+'.h5'
        else:
            loadname = lp+'checkpoint_'+mod_type+'_'+pats_ids_in[curr_fold//n_folds_sbj]+'_testday_'+str(test_day)+\
                       '_fold'+str(curr_fold//(len(pats_ids_in)))+save_suffix+'.h5'

        model_curr = tf.keras.models.load_model(loadname)

        #Grab 2D conv layer and use it to transform white noise input
        if temp_conv_layer == 'conv2d':
            output_layer = temp_conv_layer
            srate_new = srate
            if combined_sbjs:
                intermediate_layer_model = Model(inputs=[model_curr.input[0]],
                                                 outputs=[model_curr.get_layer(output_layer).output])
            else:
                intermediate_layer_model = Model(inputs=[model_curr.input],
                                                 outputs=[model_curr.get_layer(output_layer).output])
        elif temp_conv_layer == 'separable_conv2d':
            srate_new = srate/4 # srate for data into separable convolution goes through 1 average pooling
            # use longer time length for better power computation
            tLen = int(model_curr.get_layer('conv2d').input.get_shape()[-1]) 
            chLen = int(model_curr.get_layer(temp_conv_layer).input.get_shape()[-2])

            #Create model with just convolution layer
            input1   = Input(shape = (1, chLen, tLen))
            block1   = Conv2D(int(F1*D), (1, kernLength_sep), padding = 'same',
                              input_shape = (1, chLen, tLen),
                              use_bias = False,name='conv2d')(input1)
            intermediate_layer_model = Model(inputs=[input1],outputs=[block1])

            #Add in weights from separable convolution
            new_w = np.moveaxis(model_curr.get_layer(temp_conv_layer).get_weights()[0],-1,-2)
            intermediate_layer_model.get_layer('conv2d').set_weights([new_w])

        # intermediate_layer_model.summary()
        nrows = int(intermediate_layer_model.input[0].shape[-2])
        ncols = int(intermediate_layer_model.input[0].shape[-1])
        data_in = np.zeros([1,1,nrows,ncols])
        for i in range(nrows):
            data_in[...,i,:] = np.random.standard_normal(ncols)
        filt_dat = intermediate_layer_model.predict(data_in) #error pops up if running on CPU

        #Compute spectral power density for orignal and filtered dummy data
        w = hann(data_in.shape[-1])
        pow_orig = fft(data_in*w)
        pow_filt = fft(filt_dat*w)
        f = np.fft.fftfreq(pow_orig.shape[-1], d=1/srate_new)

        pow_orig = pow_orig[...,f>0]
        pow_filt = pow_filt[...,f>0]
        f = f[f>0]

        #Compute ratio of filtered power over original power
        pow_diff = np.zeros_like(pow_filt.real)
        for i in range(pow_filt.shape[1]):
            if bode_type=='mag':
                pow_diff[:,i,...] = 10*np.log10(np.abs(np.divide(pow_filt[:,i,...],pow_orig)).real)
            elif bode_type=='phase':
                pow_diff[:,i,...] = np.angle(np.divide(pow_filt[:,i,...],pow_orig),deg=True)

        # Take median across channels
        pow_diff_median = np.median(pow_diff,axis=2)

        if weight_type == 'weighted':
            # Weight pow_diff by accuracies for each temporal filter
            acc = []
            for i in range(len(np.unique(filts_left))):
                #pow_diff_median.shape[1]):
                for j in range(len(np.unique(filts_left_r))):
                    temp_filt_inds = np.nonzero(np.logical_and(filts_left==i,filts_left_r==j))[0]
                    acc.append(df['Accuracy'][temp_filt_inds].values.max()) #use max accuracy
            acc = np.asarray(acc)

        if weight_type == 'max':
            weighted_pow_median = pow_diff_median.max(axis=1)
        elif weight_type == 'weighted_percentile':
            perc_thresh = 85 #percentile threshold
            weighted_pow_median = np.zeros([1,pow_diff_median.shape[-1]])
            for i in range(pow_diff_median.shape[1]):
                perc_vals = np.percentile(pow_diff_median[0,i],perc_thresh)
                inds_perc = np.nonzero(pow_diff_median[0,i]>=perc_vals)[0]
                weighted_pow_median[0,inds_perc]+=1
            weighted_pow_median /= pow_diff_median.shape[1]
        elif weight_type == 'weighted':
            weighted_pow_median = np.zeros([1,pow_diff_median.shape[-1]])
            for i in range(pow_diff_median.shape[1]):
                weighted_pow_median[0,:]+=pow_diff_median[0,i]*acc[i] #*acc
            weighted_pow_median /= pow_diff_median.shape[1]
        elif weight_type == 'mean':
            weighted_pow_median = pow_diff_median.mean(axis=1)
        elif weight_type == 'median':
            weighted_pow_median = np.median(pow_diff_median,axis=1)

        if normalize_weighted_dat:
            weighted_pow_median /= np.max(weighted_pow_median)
        weighted_pow_medians_all.append(weighted_pow_median)
        del intermediate_layer_model
        tf.keras.backend.clear_session()
    weighted_pow_medians_all = np.squeeze(np.asarray(weighted_pow_medians_all))

    # Plot result
    sns.lineplot(x=(np.ones([weighted_pow_medians_all.shape[0],1])*np.expand_dims(f,0)).flatten(),
                 y=weighted_pow_medians_all.flatten(),ax=axes[0,0],ci='sd',color=colors_plt[0]) #,estimator=np.median)

    if bode_type=='mag':
        axes[0,0].set_ylim([-8,-3])
    else:
        axes[0,0].set_ylim([-90,90])
    axes[0,0].set_xlabel('Frequency (Hz)',fontsize=9,fontweight='normal')
    if bode_type=='mag':
        axes[0,0].set_ylabel('Power (dB)',fontsize=9,fontweight='normal')
    else:
        axes[0,0].set_ylabel('Phase ($^\circ$)',fontsize=9,fontweight='normal')
    axes[0,0].set_xlim([0,srate//2])
        
    ## Visualize spatial filter weights
    # Load param files
    file_pkl = open(lp+'param_file.pkl', 'rb')
    params_dict = pickle.load(file_pkl)
    file_pkl.close()
    n_folds = params_dict['n_folds']
    save_suffix = params_dict['save_suffix']
    combined_sbjs  = params_dict['combined_sbjs']
    n_chans_all  = params_dict['n_chans_all']
    pats_ids_in  = params_dict['pats_ids_in']
    rem_bad_chans  = params_dict['rem_bad_chans']
    dipole_dens_thresh  = params_dict['dipole_dens_thresh']

    # Find ROI x,y,z positions
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
                                                                   roi_proj_loadpath=roi_proj_lp)
    roi_pos_df = pd.read_csv(roi_proj_lp+'none_EC01_ROIcentroids_Lside.csv') #file with ROI positions
    roi_pos_good_df = roi_pos_df.iloc[good_ROIs,:]
    print("ROIs found")

    # Retain only the electrodes with nonzero data (initially padded because number of electrodes varies across subjects)
    # proj_mat_out : (len(pats_ids_in) x len(good_ROIs) x n_chans_all)
    n_chans_all = len(np.nonzero(proj_mat_out.reshape(-1,proj_mat_out.shape[-1]).mean(axis=0))[0])
    proj_mat_out = proj_mat_out[...,:n_chans_all]

    #For each spatial filter, plot absolute value of weights on Nilearn's glass brain (weighted by node removal accuracies) 
    colors_all,colorvals_all = [],[]
    for curr_fold in tqdm(range(n_folds)):
        if weight_type == 'weighted':
            #Load CSV file with accuracies
            df = pd.read_csv(lp+'df_acc_allbut1_'+str(curr_fold)+'_'+mod_type+'_'+dat_type+'.csv')
            if zero_sub_rand_accs:
                df['Accuracy'].values[df['Accuracy'].values<=rand_acc_thresh] = 0

            filts_left = [] #save filts left info to list
            for val in df['Filt_left'].tolist():
                exec('tmp='+val)
                filts_left.append(tmp[0]*10+tmp[1])
            filts_left = np.asarray(filts_left)

        #Load model
        if combined_sbjs:
            loadname = lp+'checkpoint_gen_'+mod_type+'_fold'+str(curr_fold)+save_suffix+'.h5'
        else:
            loadname = lp+'checkpoint_'+mod_type+'_fold'+str(curr_fold)+save_suffix+'.h5'
        model_curr = tf.keras.models.load_model(loadname)


        spatial_weights = copy.deepcopy(model_curr.get_layer('depthwise_conv2d').get_weights()[0])
        w_abs_accweight = np.zeros([spatial_weights.shape[0]])
        nrows, ncols = spatial_weights.shape[-2], spatial_weights.shape[-1]
        for i in range(nrows):
            for j in range(ncols):
                if weight_type == 'weighted':
                    filt_ind = np.nonzero(filts_left==(i*10+j))[0]
                    acc_val = df['Accuracy'][filt_ind].values
                else:
                    acc_val = 1 # do mean
                w_abs_accweight += np.abs(np.squeeze(spatial_weights[...,i,j]))*acc_val
        w_abs_accweight /= (nrows*ncols)

        #Generate color for each value from colormap
        cmap = get_cmap('OrRd')
        norm = Normalize(vmin=w_abs_accweight.min(), vmax=w_abs_accweight.max())
        norm2 = Normalize(vmin=vmin, vmax=vmax)
        colors,colorvals=list(),list()
        for val in w_abs_accweight.tolist():
            curr_val = norm2(norm(val))
            colors.append(cmap(curr_val)[0:3])
            colorvals.append(curr_val)
        tf.keras.backend.clear_session()
        colors_all.append(colors)
        colorvals_all.append(colorvals)

    colors_plt = np.asarray(colors_all).mean(axis=0).tolist()
    colorvals_all = np.asarray(colorvals_all).mean(axis=0)
    colors_plt = [tuple(val) for val in colors_plt]
    #Plot results onto Nilearn's glass brain
    node_size = 15
    sides_2_display = 'l'
    _plot_electrodes(roi_pos_good_df,node_size,colors_plt,axes[0,1],sides_2_display,1,
                     node_edge_colors='silver',alpha=1,edge_linewidths=.5)
    
    ## Plot difference spectrograms
    chan_ind_2_plot = 0
    for ii,loadname in enumerate(['diff_spec_eegnet_hilb_tfr.h5','diff_spec_eegnet_hilb_eeg_tfr.h5']):
        # Load power
        power = read_tfrs(spec_lp+loadname)[0]
        freq_lims = [power.freqs[0],125]

        # Plot spectrograms
        plot_spectrogram_subplots(2,2,ii+2,axes[1,ii],fig,power,
                                  [chan_ind_2_plot],epoch_times,vscale_val,
                                  freq_lims=freq_lims,y_tick_step=25,axvlines = [0],
                                  pad_val=0,cmap='RdBu_r')

    # Specific plot updates    
    axes[1,0].set_ylabel('Frequency (Hz)',fontsize=9,fontweight='normal')
    axes[1,0].set_xlabel('Time (sec)',fontsize=9,fontweight='normal')
    axes[1,1].set_xlabel('Time (sec)',fontsize=9,fontweight='normal')
    axes[0,0].set_ylabel('Power (dB)',fontsize=9,fontweight='normal')
    axes[0,0].tick_params(axis='both', labelsize=8)
    axes[0,0].spines['right'].set_visible(False)
    axes[0,0].spines['top'].set_visible(False)
    axes[1,0].tick_params(axis='both', labelsize=8)
    axes[1,1].tick_params(axis='both', labelsize=8)
    axes[0,0].set_title('(A)',fontsize=10,x=0,fontweight='bold',pad=15,color='dimgray')
    axes[0,0].text(125/2+5,-2.8,'Temporal convolution\nfrequency response',fontsize=10,
                        ha='center',fontweight='bold')
    axes[0,1].set_title('(B)',fontsize=10,x=0,fontweight='bold',pad=15,color='dimgray')
    axes[0,1].text(0.5,1.04,'Depthwise convolution\nweights',fontsize=10,
                        ha='center',fontweight='bold')
    # fig.text(0.5,0.47,'(C) Move \N{MINUS SIGN} rest difference spectrograms',fontsize=10,ha='center',fontweight='bold')
    axes[1,0].set_title('(C)',fontsize=10,x=0,fontweight='bold',pad=12,color='dimgray')
    axes[1,0].text(0,138,'ECoG difference spectrogram',fontsize=10,
                        ha='center',fontweight='bold')
    axes[1,0].text(0,128,'(n=12)',fontsize=8,
                        ha='center',fontweight='normal')
    axes[1,1].set_title('(D)',fontsize=10,x=0,fontweight='bold',pad=12,color='dimgray')
    axes[1,1].text(0,138,'EEG difference spectrogram',fontsize=10,
                        ha='center',fontweight='bold')
    axes[1,1].text(0,128,'(n=15)',fontsize=8,
                        ha='center',fontweight='normal')
    plt.setp(axes[1,0].get_xticklabels(), fontweight="normal")
    plt.setp(axes[1,0].get_yticklabels(), fontweight="normal")
    plt.setp(axes[1,1].get_xticklabels(), fontweight="normal")
    plt.setp(axes[1,1].get_yticklabels(), fontweight="normal")
    axes[1,0].set_yticks([1,25,50,75,100,125])
    axes[1,1].set_yticks([1,25,50,75,100,125])

    # Add colorbars
    add_colorbar(fig,vmin,vmax,cmap,label_name='Normalized\nValue',vert_pos=.65,horiz_pos=.8,
                 fontname='arial',tick_fontsize=8,label_fontsize=9,fontweight='normal',width=0.02,height=0.12,
                 label_pad=-15,label_y=1.5)
    vscale_val = 1.5
    add_colorbar(fig,-vscale_val,vscale_val,plt.cm.RdBu_r,label_name='dB',
                 label_fontsize=9,tick_fontsize=8,label_pad=-25,label_y=1.2,fontname='arial',
                 width=0.03,height=0.15,fontweight='normal',vert_pos=.2,horiz_pos=.85)
    return fig
        
        
def plot_overlap_trainnum(lp, dpi_plt = 300):
    """
    Plot the effects of electrode overlap and number of training participants on test accuracy.
    """
    
    # Load in results
    df_sbj_ntra = pd.read_csv(lp+'ntra_df.csv')
    df_sbj_fo = pd.read_csv(lp+'frac_overlap_df.csv')
    
    # Plot results on separate subplots
    fig,ax = plt.subplots(1,2,figsize=(7.5,2),dpi=dpi_plt,gridspec_kw={'wspace':.1})
    for mod_name in df_sbj_ntra['Models'].unique().tolist():
        g0 = sns.regplot(x='n_tra',y='Accuracy',data=df_sbj_ntra[df_sbj_ntra['Models']==mod_name],
                         scatter=False, logx=True,ax=ax[0])

    ax[0].set_xlabel('Number of Training Participants',fontsize=9)
    ax[0].set_ylabel('Test Accuracy',fontsize=9)
    ax[0].set_ylim([.45,.9])
    ax[0].set_yticks(np.arange(.5,1,.1))
    ax[0].set_xticks(np.arange(10)+1)
    ax[0].axhline(.5,c='k',linestyle='--')
    leg = ax[0].legend(df_sbj_ntra['Models'].unique().tolist(),loc='upper left',frameon=False, prop={'size': 9})
    ax[0].tick_params(axis='both', labelsize=8)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    # Move legend up slightly
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax[0].transAxes)
    yOffset = 0.07
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform = ax[0].transAxes)

    for mod_name in df_sbj_fo['Model'].unique().tolist():
        g1 = sns.regplot(x='Frac',y='Acc',data=df_sbj_fo[df_sbj_fo['Model']==mod_name],
                         scatter=False, ax=ax[1])

    ax[1].set_xlabel('Electrode Fraction Overlap',fontsize=9)
    ax[1].set_ylabel('',fontsize=9)
    ax[1].set_xlim([0.3,1])
    ax[1].set_ylim([.45,.9])
    ax[1].set_yticks(np.arange(.5,1,.1))
    ax[1].set_yticklabels(['']*len(ax[1].get_yticklabels()))
    ax[1].axhline(.5,c='k',linestyle='--')
    ax[1].tick_params(axis='both', labelsize=8)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    ax[0].set_title('(A)',fontsize=10,x=0,fontweight='bold',pad=8,color='dimgray')
    ax[0].text(5.5,.935,'Training participants v. test accuracy',fontsize=10,ha='center',fontweight='bold')
    ax[1].set_title('(B)',fontsize=10,x=0,fontweight='bold',pad=8,color='dimgray')
    ax[1].text(.65,.935,'Electrode overlap v. test accuracy',fontsize=10,ha='center',fontweight='bold')

    plt.show()
    return fig


def plot_spectrogram_subplots(n_rows,n_cols,subplot_num,ax1,fig,power_ave_masked,curr_ind,
                              epoch_times=[-2.5,2.5],vscale_val=3,freq_lims=[8,120],
                              y_tick_step = 25,axvlines = [0],pad_val=0.5,log_freq_scale=[],
                              cmap='bwr',scale_one_dir=False,fontweight='bold',xtick_vals=None,
                              axvline_w = 3):
    """
    Helper function to plot spectrograms in specific subplots.
    """
    if scale_one_dir:
        vscale_val_min = 0
    else:
        vscale_val_min = -vscale_val
    power_ave_masked.plot(curr_ind, baseline=None, colorbar=False, title="", yscale='linear', tmin=epoch_times[0], tmax=epoch_times[1],vmin=vscale_val_min,vmax=vscale_val,cmap=cmap,verbose=False,axes=ax1,show=False)
    for vals in axvlines:
        ax1.axvline(vals, linewidth=axvline_w, color="black", linestyle="--")  # event
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.setp(ax1.get_xticklabels(), fontsize=12, fontweight=fontweight) #, fontweight="bold")
    plt.setp(ax1.get_yticklabels(), fontsize=12, fontweight=fontweight) #, fontweight="bold")
    ax1.set_ylabel('Frequency (Hz)',fontsize=14, fontweight=fontweight)
    ax1.set_xlabel('Time (s)',fontsize=14, fontweight=fontweight)
    ax1.set_ylim(freq_lims[0],freq_lims[1]) #8,120)
    ax1.set_xlim([epoch_times[0]+pad_val,epoch_times[1]-pad_val])
    if len(log_freq_scale) == 0:
        y_tick_list = list(np.arange(0,freq_lims[1]+y_tick_step,y_tick_step))
        y_tick_list[0] = freq_lims[0]
        y_tick_list[-1] = freq_lims[1]
        ax1.set_yticks(y_tick_list) #[freq_lims[0],20,40,60,80,100,freq_lims[1]]) #[8,20,40,60,80,100,120])
    else:
        spaced_y_ticks = np.linspace(log_freq_scale[0],log_freq_scale[-1],len(log_freq_scale)+1)[:-1]
        half_step = np.mean(np.diff(spaced_y_ticks))/2
        y_tick_list = list(spaced_y_ticks+half_step)
        ax1.set_yticks(y_tick_list)
        log_freq_scale_str = [str(int(round(val))) for val in log_freq_scale]
        ax1.set_yticklabels(log_freq_scale_str)
    if xtick_vals is None:
        ax1.set_xticks(np.arange(epoch_times[0]+pad_val,epoch_times[1]-pad_val).tolist()) #[-2,-1,0,1,2])
    else:
        ax1.set_xticks(xtick_vals)
    ax1.spines["bottom"].set_linewidth(1.5)
    ax1.spines["left"].set_linewidth(1.5)
    for item in [fig, ax1]:
        item.patch.set_visible(False)
        
    if subplot_num//n_cols == (n_rows-1): #//n_cols][i%n_cols
        #Time info in last row
        plt.setp(ax1.get_xticklabels(), fontsize=12, fontweight=fontweight)
        ax1.set_xlabel('') #Time (s)',fontsize=14, fontweight="bold")
    else:
        ax1.set_xlabel('')
        
    if subplot_num%n_cols == 0: #(n_cols-1):
        #Frequency info in first column
        ax1.set_ylabel('') #Frequency (Hz)',fontsize=14, fontweight="bold")
        plt.setp(ax1.get_yticklabels(), fontsize=12, fontweight=fontweight)
    else:
        ax1.set_ylabel('')


def _plot_electrodes(locs,node_size,colors,axes,sides_2_display,N,node_edge_colors,alpha,edge_linewidths,marker='o'):
    """
    Low-level function that handles electrode plotting.
    """
    if N==1:
        ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                               node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,'linewidths':edge_linewidths,'marker': marker},
                               node_size=node_size, node_color=colors,axes=axes,display_mode=sides_2_display)
    elif sides_2_display=='yrz' or sides_2_display=='ylz':
        colspans=[5,6,5] #different sized subplot to make saggital view similar size to other two slices
        current_col=0
        total_colspans=int(np.sum(np.asarray(colspans)))
        for ind,colspan in enumerate(colspans):
            axes[ind]=plt.subplot2grid((1,total_colspans), (0,current_col), colspan=colspan, rowspan=1)
            ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                               node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,'linewidths':edge_linewidths,'marker': marker},
                               node_size=node_size, node_color=colors,axes=axes[ind],display_mode=sides_2_display[ind])
            current_col+=colspan
    else:
        for i in range(N):
            ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                                   node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,'linewidths':edge_linewidths,'marker': marker},
                                   node_size=node_size, node_color=colors,axes=axes[i],display_mode=sides_2_display[i])
        

def add_colorbar(f_in,vmin,vmax,cmap,width=0.025,height=0.16,horiz_pos=0.85,border_width=1.5,
                 tick_len = 0,adjust_subplots_right=0.84,label_name='',tick_fontsize=14,
                 label_fontsize=18,label_pad=15,label_y=0.6,label_rotation=0,fontweight='bold',
                 fontname='Times New Roman',vert_pos=None):
    '''
    Adds colorbar to existing plot based on vmin, vmax, and cmap
    '''
    f12636, a14u3u43 = plt.subplots(1,1,figsize=(0.01,0.01))
    im = a14u3u43.imshow(np.random.random((10,10)), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.close(f12636)
    f_in.subplots_adjust(right=adjust_subplots_right)
    if vert_pos==None:
        vert_pos = (1-height)/2
    cbar_ax = f_in.add_axes([horiz_pos, vert_pos, width, height])
    cbar = f_in.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([vmin,0,vmax])
    cbar.ax.set_yticklabels([vmin,0,vmax], fontsize=tick_fontsize,
                            weight=fontweight, fontname=fontname)
    cbar.ax.tick_params(length=tick_len)
    cbar.outline.set_linewidth(border_width)
    cbar.set_label(label_name,rotation=label_rotation,fontsize=label_fontsize,
                   weight=fontweight,labelpad=label_pad, y=label_y, fontname=fontname)
    

def plot_hyptuning_results(lp):
    """
    Plots hyperparameter tuning results (code is specific to hyperparameter tuning from manuscript)
    """
    PARAM_TO_NAME = {'F1': 'Temporal Filter Count',
                     'dropoutRate': 'Dropout Rate',
                     'kernLength': 'Kernel Length',
                     'kernLength_sep': 'Seperable Kernel Length',
                     'dropoutType': 'Dropout Type',
                     'modeltype': 'Model Type',
                     'n_estimators': '# of Estimators',
                     'max_depth': 'Maximum Depth'}
    PARAM_TO_XLIM = {'dropoutRate': (.2, .8),
                     'kernLength': (24, 136),
                     'F1': (4, 20),
                     'kernLength_sep': (24, 136),
                     'n_estimators': (25, 250),
                     'max_depth': (2, 12)}
    PARAM_TO_XTICKS = {'dropoutRate': [.2, .4, .6, .8],
                     'kernLength': [24, 52, 80, 108, 136],
                     'F1': [4, 8, 12, 16, 20],
                     'kernLength_sep': [24, 52, 80, 108, 136],
                     'n_estimators': [25, 100, 175, 250],
                     'max_depth': [2, 4, 6, 8, 10, 12]}
    delta = .001
    ylim = (.45, 1)
    yticks= [.45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1]
    eegnet_params = ['dropoutRate', 'kernLength', 'F1', 'kernLength_sep']
    categorical_params = ['dropoutType', 'modeltype']
    rf_params = ['n_estimators', 'max_depth']
    study_eegnet = joblib.load(lp+'optuna_study_[\'eegnet\', \'eegnet_hilb\'].pkl')
    study_rf = joblib.load(lp+'optuna_study_[\'rf\'].pkl')
    study_eegnet_df = pd.read_csv('eegnet_trial_data_sp.csv')
    study_rf_df = pd.read_csv('rf_trial_data_sp.csv')
    colors = sns.color_palette("hls", 8)
    color_num = 0
    plt.rcParams.update({'font.size': 7.5})
    fig, axs = plt.subplots(nrows=2, ncols=8, gridspec_kw={'wspace': 0.15, 'hspace': -.62})
    col_count = 0
    row_count = 0
    plt.suptitle('Same Participant', va='bottom', y=.7, fontsize=10)
    for param in eegnet_params:
        if col_count == 0:
            axs[row_count, col_count].set(title=PARAM_TO_NAME[param], xlim=PARAM_TO_XLIM[param], xticks=PARAM_TO_XTICKS[param], ylim=ylim, yticks=yticks)
        else:
            axs[row_count, col_count].set(title=PARAM_TO_NAME[param], xlim=PARAM_TO_XLIM[param], xticks=PARAM_TO_XTICKS[param], ylim=ylim, yticks=[])
            axs[row_count, col_count].spines['left'].set_visible(False)
        sns.regplot(x=param, y='Validation Accuracy', data=study_eegnet_df, order=2, scatter=True, scatter_kws={"color": "gray", "alpha": .25, "s": 15},
                    color=colors[color_num], ax=axs[row_count, col_count])
        if col_count == 0:
            axs[row_count, col_count].set(xlabel="")
        else:
            axs[row_count, col_count].set(ylabel="", xlabel="")
        axs[row_count, col_count].spines['right'].set_visible(False)
        axs[row_count, col_count].spines['top'].set_visible(False)
        axs[row_count, col_count].set_aspect(1.0/axs[row_count, col_count].get_data_ratio(), adjustable='box')
        color_num += 1
        col_count += 1
    for param in categorical_params:
        if param == "dropoutType":
            sns.boxplot(x=param, y='Validation Accuracy', data=study_eegnet_df, color=colors[color_num],
                          ax=axs[row_count, col_count], order=["Dropout", "SpatialDropout2D"])
            sns.swarmplot(x=param, y='Validation Accuracy', data=study_eegnet_df, color="gray", alpha=.4, s=5,
                          ax=axs[row_count, col_count], order=["Dropout", "SpatialDropout2D"])
        else:
            sns.boxplot(x=param, y='Validation Accuracy', data=study_eegnet_df, color=colors[color_num],
                        ax=axs[row_count, col_count])
            sns.swarmplot(x=param, y='Validation Accuracy', data=study_eegnet_df, color="gray", alpha=.4, s=5,
                        ax=axs[row_count, col_count])
        axs[row_count, col_count].set(ylabel="", xlabel="", title=PARAM_TO_NAME[param], ylim=ylim, yticks=[])
        axs[row_count, col_count].spines['right'].set_visible(False)
        axs[row_count, col_count].spines['top'].set_visible(False)
        axs[row_count, col_count].spines['left'].set_visible(False)
        axs[row_count, col_count].set_aspect(1.0/axs[row_count, col_count].get_data_ratio(), adjustable='box')
        color_num += 1
        col_count += 1
    for param in rf_params:
        axs[row_count, col_count].set(title=PARAM_TO_NAME[param], xlim=PARAM_TO_XLIM[param], xticks=PARAM_TO_XTICKS[param], ylim=ylim, yticks=[])
        sns.regplot(x=param, y='Validation Accuracy', data=study_rf_df, order=2, scatter=True, scatter_kws={"color": "gray", "alpha": .25, "s": 15},
                    color=colors[color_num], ax=axs[row_count, col_count])
        axs[row_count, col_count].set(ylabel="", xlabel="")
        axs[row_count, col_count].spines['right'].set_visible(False)
        axs[row_count, col_count].spines['top'].set_visible(False)
        axs[row_count, col_count].spines['left'].set_visible(False)
        axs[row_count, col_count].set_aspect(1.0 / axs[row_count, col_count].get_data_ratio(), adjustable='box')
        color_num += 1
        col_count += 1
    row_count += 1
    col_count = 0
    color_num = 0
    study_eegnet_df = pd.read_csv('eegnet_trial_data.csv')
    study_rf_df = pd.read_csv('rf_trial_data.csv')
    plt.figtext(0.5, 0.48, 'Unseen Participant', ha='center', va='bottom', fontsize=10)
    for param in eegnet_params:
        if col_count == 0:
            axs[row_count, col_count].set(xlim=PARAM_TO_XLIM[param], xticks=PARAM_TO_XTICKS[param], ylim=ylim, yticks=yticks)
        else:
            axs[row_count, col_count].set(xlim=PARAM_TO_XLIM[param], xticks=PARAM_TO_XTICKS[param], ylim=ylim, yticks=[])
            axs[row_count, col_count].spines['left'].set_visible(False)
        sns.regplot(x=param, y='Validation Accuracy', data=study_eegnet_df, order=2, scatter=True, scatter_kws={"color": "gray", "alpha": .25, "s": 15},
                    color=colors[color_num], ax=axs[row_count, col_count])
        if col_count == 0:
            axs[row_count, col_count].set(xlabel="")
        else:
            axs[row_count, col_count].set(ylabel="", xlabel="")
        axs[row_count, col_count].spines['right'].set_visible(False)
        axs[row_count, col_count].spines['top'].set_visible(False)
        axs[row_count, col_count].set_aspect(1.0/axs[row_count, col_count].get_data_ratio(), adjustable='box')
        color_num += 1
        col_count += 1
    for param in categorical_params:
        if param == "dropoutType":
            sns.boxplot(x=param, y='Validation Accuracy', data=study_eegnet_df, color=colors[color_num],
                        ax=axs[row_count, col_count], order=["Dropout", "SpatialDropout2D"])
            sns.swarmplot(x=param, y='Validation Accuracy', data=study_eegnet_df, color="gray", alpha=.4, s=5,
                          ax=axs[row_count, col_count], order=["Dropout", "SpatialDropout2D"])
        else:
            sns.boxplot(x=param, y='Validation Accuracy', data=study_eegnet_df, color=colors[color_num],
                        ax=axs[row_count, col_count])
            sns.swarmplot(x=param, y='Validation Accuracy', data=study_eegnet_df, color="gray", alpha=.4, s=5,
                          ax=axs[row_count, col_count])
        axs[row_count, col_count].set(ylabel="", xlabel="", ylim=ylim, yticks=[])
        axs[row_count, col_count].spines['right'].set_visible(False)
        axs[row_count, col_count].spines['top'].set_visible(False)
        axs[row_count, col_count].spines['left'].set_visible(False)
        axs[row_count, col_count].set_aspect(1.0/axs[row_count, col_count].get_data_ratio(), adjustable='box')
        color_num += 1
        col_count += 1
    for param in rf_params:
        axs[row_count, col_count].set(xlim=PARAM_TO_XLIM[param], xticks=PARAM_TO_XTICKS[param], ylim=ylim, yticks=[])
        sns.regplot(x=param, y='Validation Accuracy', data=study_rf_df, order=2, scatter=True, scatter_kws={"color": "gray", "alpha": .25, "s": 15},
                    color=colors[color_num], ax=axs[row_count, col_count])
        axs[row_count, col_count].set(ylabel="", xlabel="")
        axs[row_count, col_count].spines['right'].set_visible(False)
        axs[row_count, col_count].spines['top'].set_visible(False)
        axs[row_count, col_count].spines['left'].set_visible(False)
        axs[row_count, col_count].set_aspect(1.0 / axs[row_count, col_count].get_data_ratio(), adjustable='box')
        color_num += 1
        col_count += 1
    plt.figtext(0.415, 0.492, 'Eegnet', ha='center', va='bottom', fontsize=9)
    plt.figtext(0.81, 0.492, 'Random Forest', ha='center', va='bottom', fontsize=9)
    plt.figtext(0.415, 0.28, 'Eegnet', ha='center', va='bottom', fontsize=9)
    plt.figtext(0.81, 0.28, 'Random Forest', ha='center', va='bottom', fontsize=9)
    plt.show()

    
def plot_fine_tuning(root_path, model = 'eegnet_hilb',dpi_plt = 300):
    """
    Plot fine-tuning results for same and unseen modality conditions.
    """
    single_sbj_folder = 'tf_sep_per/' 
#     model_used = 'eegnet_hilb'
    tf_folders = ['tf_per_1dconv/', 'tf_depth_per/', 'tf_sep_per/', 'tf_all_per/'] # folders to view transfer learning results from
    spec_meas = ['power', 'relative_power']
    used_per_vals = True #if True, used percentage values (otherwise, used number of trials)
    nb_classes = 2 # number of label types (2 for naturalistic ECoG dataset)
    
    data_snapshot = 17.0 # Change this if you want to look at a different percentage of training data
    model_types = ['Tailored', 'Pretrain', 'Temp', 'Depth', 'Sep', 'All']
    not_finetune = 2
    n_train_default = 7
    acc_types = ['Train','Val','Test']
    # Use this to change the colors of plots
    model_palette = {"Tailored": "#c17db8B3", 
                    "Pretrain": "#2977ccB3",
                    "Temp": "#d3f0ceB3",
                    "Depth": "#9accb2B3",
                    "Sep": "#57a6b8B3",
                    "All": "#205d80B3",
                    " ": "white"}
    model_dict = {0: "Tailored",  
                 1: "Pretrain", 
                 2: "Temp", 
                 3: "Depth",
                 4: "Sep",
                 5: "All"}
    subject_marker='o'
    
    # Sets up the load paths and pulls in/sets the parameters from running the model
    
    single_sub_lp = root_path + '/' + single_sbj_folder

    fig,ax = plt.subplots(2,3,dpi=dpi_plt,figsize=(7.5,4.5),
                          gridspec_kw={'wspace':.1, 'hspace':.5})
    
    for ind, i in enumerate(['ecog','eeg']):
        if i == 'eeg':
            single_sub_lp = single_sub_lp[:-1]+'_eeg/'
            lp = [root_path+'/'+val[:-1]+'_eeg/' for val in tf_folders]
        else:
            # Separate runs for same and unseen modality fine-tuning
            lp = [root_path+'/'+val for val in tf_folders]

        # Load param file from training/testing
        file_pkl = open(root_path + '/combined_sbjs_' + spec_meas[ind] + '/param_file.pkl', 'rb')
        params_dict = pickle.load(file_pkl)
        file_pkl.close()

        rand_seed = params_dict['rand_seed']
        n_folds = params_dict['n_folds']
        n_folds = 36
        pats_ids_in = params_dict['pats_ids_in']
        n_val = params_dict['n_val']
        n_test = params_dict['n_test']
        n_train = params_dict['n_train'] 
    #     Set this to 1 so that we know default was used and np arrays will be correct dimensions
#         n_train = 1
        
        if i == 'ecog':
            n_sbjs = len(pats_ids_in)
            sbj_total_move_events_dict = {
                "EC01": 659,"EC02": 209,"EC03": 584,"EC04": 268,
                "EC05": 203,"EC06": 869,"EC07": 675,"EC08": 850,
                "EC09": 151,"EC10": 620,"EC11": 896,"EC12": 947
            }
            patIDs_sm = []
            for j in range(n_sbjs):
                patIDs_sm.append('EC' + str(j + 1).zfill(2))
            np.random.seed(rand_seed)
            # Uses the same random seed, so will create the same test subject list
            sbj_inds_all_train, sbj_inds_all_val, sbj_inds_all_test = folds_choose_subjects(n_folds, pats_ids_in,
                                                                                            n_test=n_test, n_val=n_val)
            sbj_inds_all_test_sm = [val[0] for val in sbj_inds_all_test]
            test_sbj_folds = np.asarray(sbj_inds_all_test_sm)
            
            fnames_pretrained = natsort.natsorted(glob.glob(root_path + '/'+tf_folders[0]+'/acc_gen_tf_pretrain_'+model+'*.npy'))
            # Load in accuracy values across folds
            
            
        elif i == 'eeg':
            # No param file when running the EEG dataset
            n_sbjs = 15
            n_folds = 36
            #     Set this to 1 so that we know default was used and np arrays will be correct dimensions
#             n_train = 1
            patIDs_sm = []
            for j in range(n_sbjs):
                patIDs_sm.append('EE' + str(j + 1).zfill(2))
            sbj_total_move_events_dict = {
                "EE01": 96,"EE02": 96,"EE03": 96,"EE04": 96,"EE05": 96,
                "EE06": 96,"EE07": 96,"EE08": 96,"EE09": 96,"EE10": 96,
                "EE11": 96,"EE12": 96,"EE13": 96,"EE14": 96,"EE15": 96
            }
            
            fnames_pretrained = natsort.natsorted(glob.glob(root_path + '/'+tf_folders[0][:-1]+'_eeg/acc_gen_tf_pretrain_EE01_'+model+'*.npy'))
            
        # Determine number of train and val trials available from filenames
        n_train_trials,n_val_trials = [],[]
        # Load in the percentage/number of trials from name
        for fname in fnames_pretrained:
            if used_per_vals:
                n_val_trials.append(int(fname.split('_')[-1][4:-4]))
                n_train_trials.append(int(fname.split('_')[-2][4:]))
            else:
                n_val_trials.append(int(fname.split('_')[-2][4:]))
                n_train_trials.append(int(fname.split('_')[-3][4:]))

        # Load data
        if i == 'ecog':
            accs_all_pretrain = np.zeros([n_folds,len(acc_types),len(n_val_trials),len(tf_folders)])
            accs_all_trained = accs_all_pretrain.copy()
            for k in range(len(tf_folders)):
                for j in range(len(n_val_trials)):
                    cur_load = lp[k]
                    if used_per_vals:
                        fname_curr = natsort.natsorted(glob.glob(cur_load+'acc_gen_tf_pretrain_'+model+'_'+str(n_folds)+'_ptra'+\
                                                                 str(n_train_trials[j])+'_pval'+str(n_val_trials[j])+'*.npy'))
                    else:
                        fname_curr = natsort.natsorted(glob.glob(cur_load+'acc_gen_tf_pretrain_'+model+'_'+str(n_folds)+'_ntra'+\
                                                                 str(n_train_trials[j])+'_nval'+str(n_val_trials[j])+'*.npy'))

                    # Transfer data from files to nparray
                    tmp_vals = np.load(fname_curr[0])
                    accs_all_pretrain[...,j,k] = tmp_vals
#                     accs_all_pretrain.append(tmp_vals)


                    # transfer learn accuracies
                    # Load the files
                    if used_per_vals:
                        fname_curr = natsort.natsorted(glob.glob(cur_load+'acc_gen_tf_trained_'+model+'_'+str(n_folds)+'_ptra'+\
                                                                 str(n_train_trials[j])+'_pval'+str(n_val_trials[j])+'*.npy'))
                    else:
                        fname_curr = natsort.natsorted(glob.glob(cur_load+'acc_gen_tf_trained_'+model+'_'+str(n_folds)+'_ntra'+\
                                                                 str(n_train_trials[j])+'_nval'+str(n_val_trials[j])+'*.npy'))

                    # Transfer data from files to nparray
                    tmp_vals = np.load(fname_curr[0])
                    accs_all_trained[...,j,k] = tmp_vals
#                     accs_all_trained.append(tmp_vals)
        elif i == 'eeg':
            accs_all_pretrain = np.zeros([n_folds,len(patIDs_sm),len(acc_types),len(n_val_trials),len(tf_folders)])
            accs_all_trained = accs_all_pretrain.copy()
            for k in range(len(tf_folders)):
                for j in range(len(n_val_trials)):
                    cur_load = lp[k]
                    for s,sbj in enumerate(patIDs_sm):
                        if used_per_vals:
                            fname_curr = natsort.natsorted(glob.glob(cur_load+'acc_gen_tf_pretrain_'+sbj+'_'+model+'_'+str(n_folds)+'_ptra'+\
                                                                     str(n_train_trials[j])+'_pval'+str(n_val_trials[j])+'*.npy'))
                        else:
                            fname_curr = natsort.natsorted(glob.glob(cur_load+'acc_gen_tf_pretrain_'+sbj+'_'+model+'_'+str(n_folds)+'_ntra'+\
                                                                     str(n_train_trials[j])+'_nval'+str(n_val_trials[j])+'*.npy'))

                        # Transfer data from files to nparray
                        tmp_vals = np.load(fname_curr[0])
                        if s==0:
                            pre_dat = np.expand_dims(tmp_vals.copy(),1)
                        else:
                            pre_dat = np.concatenate((pre_dat,np.expand_dims(tmp_vals.copy(),1)),axis=1)


                        # transfer learn accuracies
                        # Load the files
                        if used_per_vals:
                            fname_curr = natsort.natsorted(glob.glob(cur_load+'acc_gen_tf_trained_'+sbj+'_'+model+'_'+str(n_folds)+'_ptra'+\
                                                                     str(n_train_trials[j])+'_pval'+str(n_val_trials[j])+'*.npy'))
                        else:
                            fname_curr = natsort.natsorted(glob.glob(cur_load+'acc_gen_tf_trained_'+sbj+'_'+model+'_'+str(n_folds)+'_ntra'+\
                                                                     str(n_train_trials[j])+'_nval'+str(n_val_trials[j])+'*.npy'))

                        # Transfer data from files to nparray
                        tmp_vals = np.load(fname_curr[0])
                        if s==0:
                            tr_dat = np.expand_dims(tmp_vals.copy(),1)
                        else:
                            tr_dat = np.concatenate((tr_dat,np.expand_dims(tmp_vals.copy(),1)),axis=1)
                        
                    accs_all_pretrain[...,j,k] = pre_dat
                    accs_all_trained[...,j,k] = tr_dat
        
        # Need a separate cell for the single subject accuracies since the structure is pretty different
        accs_all_singlesbj = []
        if i == 'ecog':
            for j in range(len(n_val_trials)):
                if used_per_vals:
                    fname_curr = (single_sub_lp+'acc_gen_tf_singlesub_'+model+'_'+\
                                                             str(n_folds)+'_ptra'+str(n_train_trials[j])+'_pval'+\
                                                             str(n_val_trials[j])+'.npy')
                else:
                    fname_curr = natsort.natsorted(glob.glob(single_sub_lp+'acc_gen_tf_singlesub_['+str(sb)+']'+model+'_'+\
                                                             str(n_folds)+'_ntra'+str(n_train_trials[j])+'_nval'+\
                                                             str(n_val_trials[j])+'*.npy'))
                # Store single subject accuracies
                tmp_vals = np.load(fname_curr)
                accs_all_singlesbj.append(tmp_vals)    
            accs_all_singlesbj = np.asarray(accs_all_singlesbj)
            accs_all_singlesbj = np.moveaxis(accs_all_singlesbj,0,-1)
        elif i == 'eeg':
            accs_all_singlesbj = np.zeros([n_folds, len(patIDs_sm), len(acc_types), len(n_val_trials)])
            for j in range(len(n_val_trials)):
                for s,sbj in enumerate(patIDs_sm):
                    if used_per_vals:
                        fname_curr = (single_sub_lp+'acc_gen_tf_singlesub_'+sbj+'_'+model+'_'+\
                                        str(n_folds)+'_ptra'+str(n_train_trials[j])+'_pval'+\
                                        str(n_val_trials[j])+'.npy')
                    else:
                        fname_curr = natsort.natsorted(glob.glob(single_sub_lp+'acc_gen_tf_singlesub_'+sbj+'_'+model+'_'+\
                                       str(n_folds)+'_ntra'+str(n_train_trials[j])+'_nval'+\
                                       str(n_val_trials[j])+'*.npy'))
                    # Store the single subject accuracies
                    tmp_vals = np.load(fname_curr)
                    accs_all_singlesbj[:,s,:,j] = tmp_vals
        
        
        # Average test accuracy for each participant
        ave_vals_test = np.zeros([n_sbjs,len(model_types), len(n_val_trials)])
        test_ind = np.nonzero(np.asarray(acc_types)=='Test')[0]

        for sbj in range(n_sbjs):
        #     Get the current subjects folds (depends on dataset)
            if i=='ecog':
                folds_sbj = np.nonzero(test_sbj_folds==sbj)[0]
            else:
                folds_sbj = sbj
            for k, model_t in enumerate(model_types):
                for j in range(len(n_val_trials)):
        #             Need to separate by dataset since the structure different
                        if i=='ecog':
                            if model_t == 'Tailored':
                                ave_vals_test[sbj,k,j] = round(np.median(accs_all_singlesbj[folds_sbj,test_ind,j]),2)
                            elif model_t == 'Pretrain':
                                ave_vals_test[sbj,k,j] = round(np.median(accs_all_pretrain[folds_sbj,test_ind,j,0],axis=0),2)
                            else:
                                ave_vals_test[sbj,k,j] = round(np.median(accs_all_trained[folds_sbj,test_ind,j,k - not_finetune],axis=0),2)
                                # -2 because 2 models that are not finetuned

                        elif i=='eeg':
                            if model_t == 'Tailored':
                                ave_vals_test[sbj,k,j] = round(np.median(accs_all_singlesbj[:,folds_sbj,test_ind,j]),2)
                            elif model_t == 'Pretrain':
                                ave_vals_test[sbj,k,j] = round(np.median(accs_all_pretrain[:,sbj,test_ind,j,0]),2)
                            else:
                                ave_vals_test[sbj,k,j] = round(np.median(accs_all_trained[:,sbj,test_ind,j,k - not_finetune]),2)
                                
        # Convert to dataframe
        # creates a numpy array that says which model was used for which trial and subject
        models_np = np.zeros((len(n_train_trials) * n_sbjs * len(model_dict), 1) )
        for j in range(len(model_dict)):
        #     indexes to the proper spot for each model
            models_np[(j*(len(n_train_trials) * n_sbjs)):(j*(len(n_train_trials) * n_sbjs) + (len(n_train_trials) * n_sbjs))] = j

        # Get the data that shows what data amt in percentages
        data_amts = np.tile(n_train_trials, (n_sbjs, 1) )

        # Then calculate the actual numbers for the datapoints (should all be different when ECoG data)
        num_events_np = np.zeros( (n_sbjs, len(n_train_trials)) )
        for k,s in enumerate(patIDs_sm):
            for j,per in enumerate(n_train_trials):
                total_move_events = sbj_total_move_events_dict[s]
                num_events_np[k,j] = int(total_move_events * (per / 100))
                if i=='ecog':
        #             num_events_np just contains the move events for ECoG, so double to get total events
                    num_events_np[k,j] = num_events_np[k,j] * 2
        num_events_np = num_events_np.reshape(len(n_train_trials) * n_sbjs, 1)

        sbjs_np = np.tile(np.array(patIDs_sm).reshape(n_sbjs, 1), (len(n_train_trials))).reshape(len(n_train_trials) * n_sbjs, 1)

        # Put the pieces together in the np array for the dataframe
        all_models_test_acc_data_amts = np.empty((0,6))
        for j, model_t in enumerate(model_types):
            models_np = np.full((len(n_train_trials) * n_sbjs, 1), model_t)
            pretrain_sbjs_np = np.full((len(n_train_trials) * n_sbjs, 1), n_train_default)
    #       average test accuracy
            tmp = np.append(ave_vals_test[:,j,:].reshape(len(n_train_trials) * n_sbjs, 1), 
                    data_amts.reshape(len(n_train_trials) * n_sbjs, 1), axis = 1)
    #       number of training/finetuning events
            tmp = np.append(tmp, num_events_np, axis = 1)
    #       number of pretraining participants
            tmp = np.append(tmp, pretrain_sbjs_np, axis = 1)
    #       which subject
            tmp = np.append(tmp, sbjs_np, axis = 1)
    #       which model
            tmp = np.append(tmp, models_np, axis = 1)
            all_models_test_acc_data_amts = np.append(all_models_test_acc_data_amts, tmp, axis = 0)
            if model_t == 'Tailored':
                #       Use this to force a space between the fine tune and no finetune models in boxplot figure
                zeros = np.zeros((1, 4))
                zeros[0,1] = data_snapshot
                zeros[0,3] = n_train_default
                empty_model = [patIDs_sm[0] ,' ']
                zeros = np.append(zeros, empty_model)
                zeros = zeros.reshape((1,6))
                all_models_test_acc_data_amts = np.append(all_models_test_acc_data_amts, zeros, axis = 0)
#                 break

        # Now finally make this a dataframe!
        avg_test_acc_df = pd.DataFrame.from_records(all_models_test_acc_data_amts, columns=['Test Acc', 'Train Data Percent', 'Train Data Amt', 'Num Pretrain Sbjs', 'Subject', 'Model Type'])

        # Sets the pretrain number of training datapoints to 0 because finetuning hasn't happened yet
        # Will also help with future figures
        avg_test_acc_df.loc[avg_test_acc_df['Model Type'] == 'Pretrain', 'Train Data Amt'] = 1

        avg_test_acc_df[['Test Acc', 'Train Data Percent', 'Train Data Amt', 'Num Pretrain Sbjs']] = avg_test_acc_df[['Test Acc', 'Train Data Percent', 'Train Data Amt', 'Num Pretrain Sbjs']].apply(pd.to_numeric)
        
        #
        tmp_list = []
        if i=='ecog':
            fname_curr = natsort.natsorted(glob.glob(single_sub_lp+'acc_gen_tf_singlesub0_'+model+'_'+str(n_folds)+'*.npy'))[0]
            zero_tailored_vals = np.load(fname_curr)

        for sbj in range(n_sbjs):
        #     Get the current subjects folds (depends on dataset)
            if i=='ecog':
                folds_sbj = np.nonzero(test_sbj_folds==sbj)[0]
                sbj_median_test = round(np.median(zero_tailored_vals[folds_sbj,test_ind]),2)
                df_length = len(avg_test_acc_df)
                avg_test_acc_df.loc[df_length] = [sbj_median_test, 0, 1, n_train_default, patIDs_sm[sbj], 'Tailored']
            elif i=='eeg':
                fname_curr = natsort.natsorted(glob.glob(single_sub_lp+'acc_gen_tf_singlesub0_'+patIDs_sm[sbj]+'_'+model+'_'+str(n_folds)+'*.npy'))[0]
                zero_tailored_vals = np.load(fname_curr)
                sbj_median_test = round(np.median(zero_tailored_vals[:,test_ind]),2)
                df_length = len(avg_test_acc_df)
                avg_test_acc_df.loc[df_length] = [sbj_median_test, 0, 1, n_train_default, patIDs_sm[sbj], 'Tailored']
        

        # First plot, boxplot
        row_ind = 0 if i=='ecog' else 1
        ax[row_ind,0].axhline(1/nb_classes,c='k',linestyle='--')
        sns.boxplot(x = 'Model Type', y = 'Test Acc', 
                    data=(avg_test_acc_df.loc[(avg_test_acc_df['Train Data Percent'] == data_snapshot) & (avg_test_acc_df['Num Pretrain Sbjs'] == n_train_default) ] ) , 
                         palette = model_palette, showfliers=False,whis=0,ax=ax[row_ind,0])
        palette = sns.color_palette("Paired")
        # Adds the individual subjects on top
        swarm = sns.swarmplot(x = 'Model Type', y = 'Test Acc', hue='Subject',
                        data=(avg_test_acc_df.loc[(avg_test_acc_df['Train Data Percent'] == data_snapshot) & (avg_test_acc_df['Num Pretrain Sbjs'] == n_train_default)]), 
                        color ='black', marker=subject_marker, s=4,ax=ax[row_ind,0])
        swarm.legend_.remove()

        ax[row_ind,0].set_xlabel(' ')
        ax[row_ind,0].set_ylabel('Test Accuracy', fontsize='11')
        ax[row_ind,0].set_ylim([(1/nb_classes)-.05,1.065])
        ax[row_ind,0].set_yticks([.5,.75,1])
        ax[row_ind,0].set_xticks([0, 2, 2, 3, 4, 5, 6])
        ax[row_ind,0].spines['right'].set_visible(False)
        ax[row_ind,0].spines['top'].set_visible(False)
        ax[row_ind,0].spines['left'].set_bounds((1/nb_classes)-.05, 1)
        ax[row_ind,0].tick_params(axis='both', labelsize=8)
        ax[row_ind,0].set_xticklabels(ax[row_ind,0].get_xticklabels(), rotation = 40, ha="center")
        ax[row_ind,0].tick_params(axis='x', pad=0)
        if i=='ecog':
            ax[row_ind,0].set_title('(A)',fontsize=10,fontweight='bold',pad=8,color='dimgray', x=0)
        else: 
            ax[row_ind,0].set_title('(D)',fontsize=10,fontweight='bold',pad=8,color='dimgray', x=0)

#         # Draw the significance on
#         star_fontsize = 18
#         if i=='ecog':
#             y_start,h,w_x,top_bar,x_start = 0.97,.04,1.9,4,2
#             ax[row_ind,0].plot([x_start, x_start, top_bar, top_bar],
#                       [y_start, y_start+h, y_start+h, y_start],
#                       lw=1.5, c='k')
#             ax[row_ind,0].text(0.75*top_bar,y_start+0.02,'*',fontsize=star_fontsize,fontweight='bold',ha='center')

#         else:
#             y_start,h,w_x,top_bar,x_start = 1.02,.04,0.9,5.5, 0
#             ax[row_ind,0].plot([x_start, x_start, top_bar, top_bar, top_bar-w_x, top_bar+w_x],
#                       [y_start, y_start+h, y_start+h, y_start, y_start, y_start],
#                       lw=1.5, c='k')
#             ax[row_ind,0].text(0.5*top_bar,y_start+0.02,'*',fontsize=star_fontsize,fontweight='bold',ha='center')

#             y_start,h,w_x,top_bar,x_start = 0.94,.04,1.9,4.5,2
#             ax[row_ind,0].plot([x_start, x_start, top_bar, top_bar, top_bar-w_x, top_bar+w_x],
#                       [y_start, y_start+h, y_start+h, y_start, y_start, y_start],
#                       lw=1.5, c='k')
#             ax[row_ind,0].text(0.7*top_bar,y_start+0.02,'**',fontsize=star_fontsize,fontweight='bold',ha='center')

        # Second plot, individual subject accuracies
        sns.lineplot(x='Subject',y='Test Acc',hue='Model Type',
                     data=(avg_test_acc_df.loc[(avg_test_acc_df['Train Data Percent'] == data_snapshot) \
        #                                        & (avg_test_acc_df['Model Type'].isin(models_use))
                                               & (avg_test_acc_df['Num Pretrain Sbjs'] == n_train_default) ] ),
                     ax=ax[row_ind,1],marker='o',markersize=5,linewidth=1,
                     palette=model_palette, ci=None)
        leg = ax[row_ind,1].legend()
        leg_lines = leg.get_lines()
        
        for j in range(len(model_dict)+1):
            ax[row_ind,1].lines[j].set_linestyle("None") 
        #     leg_lines[j].set_linestyle("None") 
        ax[row_ind,1].legend_.remove()

        ax[row_ind,1].axhline(1/nb_classes,c='k',linestyle='--')
        ax[row_ind,1].set_ylim([(1/nb_classes)-.05,1.065])
        ax[row_ind,1].set_yticks([.5,.75,1])
        ax[row_ind,1].set_yticklabels([])
        ax[row_ind,1].set_ylabel('', fontsize='11')
        ax[row_ind,1].spines['right'].set_visible(False)
        ax[row_ind,1].spines['top'].set_visible(False)

        for tick in ax[row_ind,1].get_xticklabels():
            tick.set_rotation(60)
            tick.set_fontsize(8)
        ax[row_ind,1].tick_params(axis='x', pad=0)
        ax[row_ind,1].set_xlabel(' ')
        ax[row_ind,1].spines['left'].set_bounds((1/nb_classes)-.05, 1)
        if i=='ecog':
            ax[row_ind,1].set_title('(B)',fontsize=10,fontweight='bold',pad=8,color='dimgray', x=0)
        else: 
            ax[row_ind,1].set_title('(E)',fontsize=10,fontweight='bold',pad=8,color='dimgray', x=0)


        # Last plot, finetuning curves
        models_use = ["Tailored", "Temp", "Depth", "Sep", "All"]
        # Replaces some of the pretrain models with all so we can see what happens at 0 finetuning events
        avg_test_acc_df.loc[(avg_test_acc_df['Model Type'] == 'Pretrain') 
                            & (avg_test_acc_df['Train Data Percent'] == 17.0), "Model Type"] = 'All'
        avg_test_acc_df.loc[(avg_test_acc_df['Model Type'] == 'Pretrain') 
                            & (avg_test_acc_df['Train Data Percent'] == 67.0), "Model Type"] = 'Temp'
        avg_test_acc_df.loc[(avg_test_acc_df['Model Type'] == 'Pretrain') 
                            & (avg_test_acc_df['Train Data Percent'] == 33.0), "Model Type"] = 'Depth'
        avg_test_acc_df.loc[(avg_test_acc_df['Model Type'] == 'Pretrain') 
                            & (avg_test_acc_df['Train Data Percent'] == 50.0), "Model Type"] = 'Sep'

        for m, model_t in enumerate(models_use):
            g0 = sns.regplot(x="Train Data Amt", y = "Test Acc",
                   data = avg_test_acc_df.loc[(avg_test_acc_df['Model Type']==model_t) 
                                              & (avg_test_acc_df['Num Pretrain Sbjs'] == n_train_default)],
                            scatter=False, scatter_kws={"s": 12, 'alpha':0.5}, logx=True, ax=ax[row_ind,2], color = model_palette[model_t]) 


        ax[row_ind,2].spines['left'].set_bounds((1/nb_classes)-.05, 1)
        ax[row_ind,2].set_ylim([(1/nb_classes)-.05,1.065])
        ax[row_ind,2].set_yticks([.5,.75,1])
        ax[row_ind,2].set_yticklabels([])
        ax[row_ind,2].set_ylabel(" ")
        ax[row_ind,2].spines['right'].set_visible(False)
        ax[row_ind,2].spines['top'].set_visible(False)
        ax[row_ind,2].axhline(1/nb_classes,c='k',linestyle='--')
        if i=='ecog':
            ax[row_ind,2].set_xticks([0, 400, 800, 1200])
            ax[row_ind,2].set_xticklabels([0, 400, 800, 1200], fontsize='8')
            ax[row_ind,2].set_xlabel("")
            ax[row_ind,2].set_xlim(-20, 1300)
            ax[row_ind,2].legend(title = 'Model Type', title_fontsize = 'x-small', labels=models_use, bbox_to_anchor=(0.94, 0.6), prop={'size':6}, markerscale=1, frameon = False)
        else:    
            ax[row_ind,2].set_xticks([1, 21, 41, 61])
            ax[row_ind,2].set_xticklabels([0, 20, 40, 60], fontsize='8')
            ax[row_ind,2].set_xlim(-1, 65)
            ax[row_ind,2].set_xlabel("Training Events", fontsize='8')

        if i=='ecog':
            ax[row_ind,2].set_title('(C)',fontsize=10,fontweight='bold',pad=8,color='dimgray', x=0)
        else: 
            ax[row_ind,2].set_title('(F)',fontsize=10,fontweight='bold',pad=8,color='dimgray', x=0)
            
    return fig, avg_test_acc_df


def test_finetuning_stats(avg_test_acc_df, model_types = ['Tailored', 'Pretrain', 'Temp', 'Depth', 'Sep', 'All']):
    """
    Compute non-parametric stats for fine-tuned decoders.
    """
    print(pg.friedman(data=avg_test_acc_df.loc[(avg_test_acc_df['Model Type'].isin(model_types)) 
                                          & (avg_test_acc_df['Train Data Percent'] == data_snapshot)], 
                      dv='Test Acc', within='Model Type', subject='Subject')['p-unc'])
    # Otherwise can just look at the whole dataset
    print(pg.friedman(data=avg_test_acc_df.loc[(avg_test_acc_df['Model Type'].isin(model_types))], dv='Test Acc', within='Model Type', subject='Subject')['p-unc'])
    # Looks like it gives significant results

    # Wilcoxon tests (non-parametric t-tests)
    # Computes tests at the data snapshot
    p_vals = []
    n_models = len(model_types)
    for i in range(n_models):
        for j in range(i+1,n_models):
            val1 = avg_test_acc_df[(avg_test_acc_df['Model Type'] == model_types[i]) & 
                                   (avg_test_acc_df['Train Data Percent'] == data_snapshot) &
                                   (avg_test_acc_df['Num Pretrain Sbjs'] == n_train_default)].iloc[:,0].values
            val2 = avg_test_acc_df[(avg_test_acc_df['Model Type'] == model_types[j]) & 
                                   (avg_test_acc_df['Train Data Percent'] == data_snapshot)&
                                   (avg_test_acc_df['Num Pretrain Sbjs'] == n_train_default)].iloc[:,0].values
            p_vals.append(float(pg.wilcoxon(val1, val2)['p-val']))

    # Correct for multiple comparisons
    _,p_vals = pg.multicomp(np.asarray(p_vals), alpha=0.05, method='fdr_bh')

    pval_df = np.zeros([n_models,n_models])
    q = 0
    for i in range(n_models):
        for j in range(i+1,n_models):
            pval_df[i,j] = p_vals[q]
            q += 1

    # Create output df with p_values
    df_pval = pd.DataFrame(pval_df,columns=model_types,index=model_types)
    print(df_pval)
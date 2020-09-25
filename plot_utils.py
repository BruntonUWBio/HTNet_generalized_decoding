"""
Utility functions for all Jupyter notebook plots.
"""
from sklearn.externals import joblib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
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
    file_pkl = open(root_path+dataset+'/combined_sbjs/param_file.pkl', 'rb')
    params_dict = pickle.load(file_pkl)
    file_pkl.close()

    rand_seed = params_dict['rand_seed']
    n_folds = params_dict['n_folds']
    pats_ids_in = params_dict['pats_ids_in']
    combined_sbjs = params_dict['combined_sbjs']
    test_day = params_dict['test_day']
    n_test = params_dict['n_test']
    n_val = params_dict['n_val']
    lp = [root_path+dataset+suffix_lp]

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
                    lp_curr = lp_curr.replace(replace_str,'')
                    mod_type_curr = '' if model_type=='eegnet_hilb' else '_'+model_type
                    if not compare_models:
                        lp_spl = lp_curr.split('/')
                        lp_spl[:-2]+lp_spl[-1:]
                        lp_curr = '/'.join(lp_spl[:-2]+lp_spl[-1:])
                        suff = lp_spl[-2]
                    else:
                        suff = ''
                        ##########Use relative power instead for comparison##########
                        if mod_type_curr=='':
                            suff = '_relative_power'
                        ##########Use relative power instead for comparison##########
                    tmp_vals = np.load(lp_curr+'accs_ecogtransfer'+mod_type_curr+suff+'.npy')
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
                    if len(pat_curr) > 3:
                        pat_curr = pat_curr[:3]
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
        # Add stats (manually created)
        star_fontsize = 18
        if (ii==0) & compare_models:
            y_start,h,w_x = 1.02,.04,1
            ax[0,ii].plot([0, 0, 2, 2, 2-w_x, 2+w_x],
                          [y_start, y_start+h, y_start+h, y_start, y_start, y_start],
                          lw=1.5, c='k')
            if use_asterisks:
                ax[0,ii].text(1,1.04,'*',fontsize=star_fontsize,fontweight='bold',ha='center')
            else:
                ax[0,ii].text(1,1.09,r'$p<0.05$',fontsize=7,fontweight='normal',ha='center')
        elif (ii==1) & compare_models:
            y_start,h1,h2,w_x = .95,.06,.06,1
            ax[0,ii].plot([0, 0, 2, 2, 2-w_x, 2+w_x],
                          [y_start, y_start+h1, y_start+h1, y_start-h2, y_start-h2, y_start-h2],
                          lw=1.5, c='k')
            if use_asterisks:
                ax[0,ii].text(1,.99,'*',fontsize=star_fontsize,fontweight='bold',ha='center')
            else:
                ax[0,ii].text(1,1.05,r'$p<0.05$',fontsize=7,fontweight='normal',ha='center')
        elif (ii==2) & compare_models:
            y_start,h1,h2,w_x = .8,.06,.1,1
            ax[0,ii].plot([0, 0, 2, 2, 2-w_x, 2+w_x],
                          [y_start, y_start+h1, y_start+h1, y_start-h2, y_start-h2, y_start-h2],
                          lw=1.5, c='k')
            if use_asterisks:
                ax[0,ii].text(1,.85,'***',fontsize=star_fontsize,fontweight='bold',ha='center')
            else:
                ax[0,ii].text(1,.9,r'$p<0.001$',fontsize=7,fontweight='normal',ha='center')
        if (ii==0) & (not compare_models):
            ax[0,ii].text(2,.82,'**',fontsize=star_fontsize,fontweight='bold',ha='center')
            ax[0,ii].text(3,.97,'**',fontsize=star_fontsize,fontweight='bold',ha='center')
        elif (ii==1) & (not compare_models):
            y_start,h = .92,.06
            ax[0,ii].plot([2, 2, 4, 4],
                          [y_start, y_start+h, y_start+h, y_start],
                          lw=1.5, c='k')
            ax[0,ii].text(3,.97,'**',fontsize=star_fontsize,fontweight='bold',ha='center')
        elif (ii==2) & (not compare_models):
            ax[0,ii].text(1,.55,'**',fontsize=star_fontsize,fontweight='bold',ha='center')
            ax[0,ii].text(2,.8,'**',fontsize=star_fontsize,fontweight='bold',ha='center')
            ax[0,ii].text(3,.65,'**',fontsize=star_fontsize,fontweight='bold',ha='center')

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
            leg_lines[i+1].set_linestyle("None") # 'None') #
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
    
def test_acc_mod_states(dfs_all):
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
            loadname = lp+'checkpoint_'+mod_type+'_'+pats_ids_in[curr_fold//n_folds_sbj][:3]+'_testday_'+str(test_day)+\
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
    roi_pos_df = pd.read_csv(roi_proj_lp+'none_a0f66459_ROIcentroids_Lside.csv') #file with ROI positions
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
    ax[1].set_yticklabels(['','','','','',''])
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
    power_ave_masked.plot(curr_ind, baseline=None, colorbar=False, title="", yscale='linear', tmin=epoch_times[0], tmax=epoch_times[1],vmin=vscale_val_min,vmax=vscale_val,cmap=cmap,verbose=False,axes=ax1)
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

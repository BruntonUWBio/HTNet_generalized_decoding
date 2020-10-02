# HTNet_generalized_decoding

Code for HTNet as shown in [PAPER LINK]. The HTNet model code is available in *htnet_model.py*. This model is written in Python and relies on Keras and Tensorflow. It is heavily based on the EEGNet model developed by Lawhern et al: https://github.com/vlawhern/arl-eegmodels.

To replicate the findings from our paper, create a new conda environment using the environment.yml file. Be sure to download the data from figshare [DATA LINK].



**Run all decoder training and testing analyses**

Open *train_decoders.py* and set rootpath to be the directory above your ecog_dataset and eeg_dataset directories and then run it. Note that this script will run every analysis from our paper at once, which takes several days to run. The different analyses are separated out in the script in case you want to comment out certain ones. This script also requires a GPU (change integer for os.environ["CUDA_VISIBLE_DEVICES"] if you need to switch to a different GPU).



**Plot results**

Once *train_decoders.py* has finished runnning, open *Plot_figures.ipynb* and add the rootpath directory you used previously. Each cell will produce a plot similar to the figures in the paper. Note that the HTNet interpretability cell requires a GPU to compute the frequency response of the temporal convolution.

As a side note, the default images were generated using models trained for only 2 epochs as a test, which is why they do not match our manuscript figures.



**Hyperparameter tuning**

Additionally, we have included our hyperparameter tuning code (*hyptuning_optuna.py*), which uses Optuna to tune hyperparameters.

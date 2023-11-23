# Description of the (4th public, 1st private) solution 

## TL;DR

+ Unet, Linknet with vgg16 and vgg19 backbones (segmentation-models library).
+ Input: 2 or 4 consecutive radar scans (precipitation intensity). Output: next radar scan.
+ Prediction for 12 timesteps is done in a consecutive manner by predicting one scan at time and then adding it to the input sequence for further prediction. 
+ Log-transform of data.
+ Rotations (90, 180, 270).
+ Training on different train/validation splits.
+ Learning rate reduction.
+ Three successful model tiers which differ by utilized models and backbones and different learning rate policy. 
+ After evaluation of each tier: taking maximum of 3 best model predictions.
+ Final ensembles: mean of calculated tiers' maximums (157.02 on public, 164.92 on private).

## Files and folders overview

+ models/ - placeholder for storing keras models in case of re-training.
+ models_final/ - keras models that have been used for final solution.
+ outputs/ - reproduced output files based on final models, but in a new environment and using CPU.
+ outputs_final/ - output files that have been uploaded to the assessment system.
+ train/ - hdf5 files provided by Yandex.
+ train_npy/ - folder for individual .npy files that holds numpy arrays with precipitation intensity (see also "02_hdf2npy.ipynb")

+ 01_get_keys.ipynb - helper for established training routine.
+ 02_hdf2npy.ipynb - conversion of data from hdf to npy for allowing multiprocessing of IO tasks.
+ 03_eval1000.ipynb - helper for faster evaluation of trained models on a random subset of validation.
+ 04_tier3.py, 05_tier5.py, 06_tier7.py - scripts for training and evaluation of models (see further step-by-step description).
+ 07_get_output.py - generation of output files based on trained models and provided test dataset.
+ 08_average_best_models.ipynb - notebook with utilized ensembles.
+ 09_checking_repr.ipynb - notebook that checks reproducibility of the proposed routine by comparing the proximity of newly generated outputs (folder "outputs/") with ones submitted to the evaluation system (folder "outputs_final/").
+ 2022-test-public.hdf5 - test file provided by Yandex.
+ env_yacup_tf.yml - conda/mamba environment file.
+ eval1000.npy - keys for fast evaluation of trained models.
+ keys.npy - all available keys that grouped alongside different sequence lengths to help data splitting. 


## Trained models and submitted outputs

+ [Link](https://mega.nz/file/jFkx1LLB#DDZN-sMASEJY_VqUVFjBh3b_wM0xufh0w5j0SyjZq1s) for downloading final versions of trained models (folder models_final) and output files (folder outputs_final), 2.52 Gb in total.
+ [Everything in one archive](https://mega.nz/file/PNEl2KKS#T1f3SQlRzhb2rABnaoKLz9WBB3iSyLuJKw52wSf1VU4) (as of November 15, 2023; 7.49 Gb).


## Step-by-step description

### Training keys

To support model training routine, the file (keys.npy) that holds all the needed key sequences (input+output) has been prepared. The structure of file follows the structure of python dictionary where first-order keys are calender months, second-order keys are considered sequence lengths (of 3, 5, and 16 timesteps), and values are the respective timestamps.

The code for this step is provided in the Jupyter notebook "01_get_keys.ipynb".


### Training data conversion

Based on experience, it was decided to convert all precipitation intensity data in the provided .hdf5 files to individual .npy files to allow their simultaneous read by many workers during training procedure. The values of -1e6 (no rain) in the hdf5 files have been changed to 0; the values of -2e6 (no measurements) to -1. The data is stored in "train_npy" folder.

The code for this step is provided in the Jupyter notebook "02_hdf2npy.ipynb".


### TIERS

The concept of tiers includes training of the particular set of models with some common hypothesis. There were 9 models tiers, among which only three have been used as a basis for a final solution, i.e. Tier 3, Tier 5, and Tier 7.

There are two common places among all the tiers: trainsets and quick final validation.

Trainset consists of two lists that include the calender months that will be used for model training and validation. There are 31 trainsets that combine different variation of 9 calendar months for training and 3 for validation. Trainset with an alias of "999" describes one that utilizes all available data for training with 4 months (2,5,8,11) for validation.

To speed-up the validation of trained models, 1000 randomly-selected sequences of length 16 have been taken from months (2,5,8,11) using the "keys.npy" file. Unfortunately, the random seed for generating evaluation keys has not been fixed, so the creation of file "eval1000.npy" could not be reproducible. The code for this step is provided in the Jupyter notebook "03_eval1000.ipynb".


#### Tier 3

Model configuration list:
+ Unet with vgg16.
+ 10 different trainsets (7, 10, 13, 16, 19, 21, 24, 27, 29, 999).
+ Training is set for 20 epochs with LR scheduler (ReduceLROnPlateau: factor=0.1, patience=2).
+ Batch size is 32.

The training routine is described in file "04_tier3.py".

The list of the best three models ({model_name}_{backbone}\_{lead_time}_{trainset}) on a public leaderboard with submission ID and respective score:

1. Unet_vgg16_1_21_tier3.keras:     96371767        158.79
2. Unet_vgg16_1_24_tier3.keras:     96372038        159.36
3. Unet_vgg16_1_19_tier3.keras:     96371117        159.48


**The output hdf5 file for every submission is generated by file "07_get_output.py" based on the saved models in .keras format (full model description and weights).**


#### Tier 5

Here, we are testing an idea of accounting more input data for model training: 4 recent radar scans instead of 2 in Tier 3.

+ Same Unet with vgg16 backbone architecture as in Tier 3.
+ 10 trainsets (11, 19, 20, 21, 22, 23, 24, 25, 26, 27).
+ 20 epochs with reduction of the learning rate (ReduceLROnPlateau: factor=0.1, patience=2).


The list of the best 2 models ({model_name}_{backbone}\_{lead_time}_{trainset}) with the respective score on a public leaderboard:

1. Unet_vgg16_1_23_tier5.keras:       96831553        159.43
2. Unet_vgg16_1_19_tier5.keras:       96829789        159.53


The code for model training is in the file "05_tier5.py".

#### Tier 7

+ Similar to the Tier 3.
+ Unet, Linknet as the core models with bigger vgg19 backbone.
+ 10 trainsets (11, 19, 20, 21, 22, 23, 24, 25, 26, 27).
+ 20 epochs with reduction of the learning rate (ReduceLROnPlateau: factor=0.1, patience=2).


The list of the best 3 models ({model_name}_{backbone}\_{lead_time}_{trainset}) with the respective score on a public leaderboard:

1. Linknet_vgg19_1_27_tier7.keras:      96824439        159.50
2. Unet_vgg19_1_24_tier7.keras:         96772099        159.67
3. Linknet_vgg19_1_25_tier7.keras:      96826719        159.72


The code for model training is in the file "06_tier7.py".


### Output file generation

The output hdf5 file for every submission is generated by file "07_get_output.py" based on the saved models in .keras format (full model description and weights, located in folder "models_final").


### Ensembles

The important part is ensembling of model results. It seemed beneficial to take maximum of the best model predictions from particular Tier. This way, we take a maximum of: 
1. Three best model predictions (on a public leaderboard) from Tier3,
2. Two best model predictions from Tier 5,
3. Three best model predictions from Tier 7.

Getting this "maxes", we average them getting the final submission result.

The code for averaging is in the file "08_average_best_models.ipynb".


### Reproducibility

To check to which extent new predictions (folder "outputs") are reproducible compared to the submitted versions (folder "outputs_final"), use "09_checking_repr.ipynb" notebook. Different math-related libraries and using "float16" could lead to some deviations. However, the issue seems to be very minor with the biggest differences (around 0.01-0.04 mm/h) detected for the minority of pixels near radar scan boundaries.


### Anaconda environment

The environment file for the installation of all required libraries is in "env_yacup_tf.yml". We used mamba for installation.
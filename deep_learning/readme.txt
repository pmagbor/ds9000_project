This directory contained the scriptts used to train the evaluate the 3D ResNet-18 model from MONAI on the MOTUM dataset.

#### MRI preprocessing 
*mri_preprocessing_3d_simple.py*: Used to implement the intensity correction, MRI brsin registration to a standard Brain templete, volume tranformation etc. Script was obtained from BrainIAC (https://github.com/AIM-KannLab/BrainIAC/tree/main). 

*train_t1_3d_train_val_test*: Script used to train 3D ResNet-18 on the T1 only form all patients included in the analysis. Training data was split into train, val, and test. Evaluation of the model was also performed on in the script and performance metrices where computed. 

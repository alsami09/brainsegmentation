import pandas as pd 
import numpy as np
import shutil
import os
import glob
import random


random.seed(306)
file_name = pd.DataFrame()
file_name_MASK= pd.DataFrame()
file_name['file name'] = glob.glob('D:\Task01_BrainTumour\imagesTr/*')
file_name['mask name'] = glob.glob('D:\Task01_BrainTumour\labelsTr/*')


msk = np.random.rand(len(file_name)) < 0.8
train_3D = file_name[msk]
test_3D = file_name[~msk]



source_dir = 'D:\Task01_BrainTumour\imagesTr'
source_dir_Mask = 'D:\Task01_BrainTumour\labelsTr' 

target_dir_Train_3D_Image = 'D:\Multiclass-Segmentation-in-Unet-master\Multiclass-Segmentation-in-Unet-master\Dataset\Train\IMAGE'
target_dir_Test_3D_Image = 'D:\Multiclass-Segmentation-in-Unet-master\Multiclass-Segmentation-in-Unet-master\Dataset\Test\IMAGE'

target_dir_Train_3D_MASK = 'D:\Multiclass-Segmentation-in-Unet-master\Multiclass-Segmentation-in-Unet-master\Dataset\Train\MASK'
target_dir_Test_3D_MASK = 'D:\Multiclass-Segmentation-in-Unet-master\Multiclass-Segmentation-in-Unet-master\Dataset\Test\MASK'


for i, file_names in train_3D.iterrows():
    try:
        shutil.move(os.path.join(source_dir, file_names['file name']), target_dir_Train_3D_Image)
    except: pass

for i, file_names in test_3D.iterrows():
    try:
        shutil.move(os.path.join(source_dir, file_names['file name']), target_dir_Test_3D_Image)
    except: pass

for i, file_names in train_3D.iterrows():
    try:
        shutil.move(os.path.join(source_dir_Mask, file_names['mask name']), target_dir_Train_3D_MASK)
    except: pass

for i, file_names in test_3D.iterrows():
    try:
        shutil.move(os.path.join(source_dir_Mask, file_names['mask name']), target_dir_Test_3D_MASK)
    except: pass


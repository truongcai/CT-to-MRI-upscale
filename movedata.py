import shutil
import os
import random
import math
import numpy as np
import argparse
from scantree import scantree, RecursionFilter
#This script will do the data set up properly in the right folders with the original dataset being in a directory outside of where this file is stored.
parser = argparse.ArgumentParser()
parser.add_argument('--testsplit', type=float, default=0.2, 
                    help='proportion of test samples')
parser.add_argument('--validsplit', type=float, default=0.2, 
                    help='proportion of validation samples')
parser.add_argument('--dataset', type=str, default='Dataset', 
                    help='name of dataset')
parser.add_argument('--batchsize', type=int, default='4', 
                    help='batchsize')
parser.add_argument('--dataroot', type=str, default='/home/nasheath_ahmed/', 
                    help='the root where you want the data to be stored')
options = parser.parse_args()

# Creating the output directories if they don't exist. 
if not os.path.exists(options.dataroot+ 'C2M_test/train/CT'):
        os.makedirs(options.dataroot+ 'C2M_test/train/CT')
if not os.path.exists(options.dataroot+ 'C2M_test/train/MR'):
        os.makedirs(options.dataroot+ 'C2M_test/train/MR')
if not os.path.exists(options.dataroot+ 'C2M_test/valid/CT'):
        os.makedirs(options.dataroot+ 'C2M_test/valid/CT')
if not os.path.exists(options.dataroot+ 'C2M_test/valid/MR'):
        os.makedirs(options.dataroot+ 'C2M_test/valid/MR')
if not os.path.exists(options.dataroot+ 'C2M_test/test/CT'):
        os.makedirs(options.dataroot+ 'C2M_test/test/CT')
if not os.path.exists(options.dataroot+ 'C2M_test/test/MR'):
        os.makedirs(options.dataroot+ 'C2M_test/test/MR')

#the directories for where the images will be. 
train_ct = options.dataroot+ 'C2M_test/train/CT/'
train_mr = options.dataroot+ 'C2M_test/train/MR/'
valid_ct = options.dataroot+ 'C2M_test/valid/CT/'
valid_mr = options.dataroot+ 'C2M_test/valid/MR/'
test_ct = options.dataroot+ 'C2M_test/test/CT/'
test_mr = options.dataroot+ 'C2M_test/test/MR/'


#Getting all the paths for the files for both CT and MR images into an array
mr_source = scantree('../Dataset/images/testB', RecursionFilter(match=['*.png', '*.jpg']))
mr_images =  [path.real for path in mr_source.filepaths()]
mr_source_train = scantree('../Dataset/images/trainB', RecursionFilter(match=['*.png', '*.jpg']))
mr_images +=  [path.real for path in mr_source_train.filepaths()]
ct_source = scantree('../Dataset/images/testA', RecursionFilter(match=['*.png', '*.jpg']))
ct_images =  [path.real for path in ct_source.filepaths()]
ct_source_train = scantree('../Dataset/images/trainA', RecursionFilter(match=['*.png', '*.jpg']))
ct_images +=  [path.real for path in ct_source_train.filepaths()]

#Storing all the images in respective folders with respect to being divisible by the batchsize of training. 
min_number = min(len(ct_images), len(mr_images))
CTdata = ct_images[:min_number]
MRdata = mr_images[:min_number]
num_test_samples = math.ceil(len(CTdata)*options.testsplit)
num_test_samples = num_test_samples - (num_test_samples % options.batchsize)
num_val_samples = math.ceil(len(CTdata)*options.validsplit)
num_val_samples = num_val_samples - (num_val_samples % options.batchsize)
count = 0
capped  = len(CTdata) - (len(CTdata) % options.batchsize)
for x in range(len(CTdata)):
        if count < num_test_samples:
                shutil.copy(CTdata[x], test_ct + 'ct' + f'{count:04}' + '.png')
                shutil.copy(MRdata[x], test_mr + 'mr' + f'{count:04}' + '.png')
        elif (count < num_test_samples*2):
                shutil.copy(CTdata[x], valid_ct + 'ct' + f'{count:04}'+ '.png')
                shutil.copy(MRdata[x], valid_mr + 'mr' + f'{count:04}'+ '.png')
        elif count < capped:
                shutil.copy(CTdata[x], train_ct + 'ct' + f'{count:04}'+ '.png')
                shutil.copy(MRdata[x], train_mr + 'mr' + f'{count:04}'+ '.png')
        count += 1

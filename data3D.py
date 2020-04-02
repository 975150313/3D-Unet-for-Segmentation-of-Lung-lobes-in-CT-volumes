from __future__ import print_function

import glob
import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from skimage.io import imread
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import nrrd
import os
from numpy.random import randint

data_path = './'

def get_filenames():   
    train_data_path = os.path.join(data_path, 'volume/')
    mask_data_path = os.path.join(data_path, 'mask/')
    train_dirs = os.listdir(train_data_path)
    mask_dirs = os.listdir(mask_data_path)
    return train_dirs, mask_dirs


def create_patch_dataset():
    
    patch_folder_scan = './patch_train_dataset/'
    patch_folder_mask = './patch_train_mask_dataset/'
    
    if not os.path.exists(patch_folder_scan):
        os.makedirs(patch_folder_scan)
    if not os.path.exists(patch_folder_mask):
        os.makedirs(patch_folder_mask)
    
    
    train_files, mask_files = get_filenames()
    train_data_path = os.path.join(data_path, 'volume/')
    mask_data_path = os.path.join(data_path, 'mask/')
    
    train_arrays_list = []
    patient_list_img = []
    mask_arrays_list = []  
    patient_list_mask = []
    for image in range(len(train_files)):
        img_train = sitk.ReadImage(train_data_path+train_files[image])
        out_img = sitk.GetArrayFromImage(img_train)
        img_mask = sitk.ReadImage(mask_data_path+mask_files[image])
        out_mask = sitk.GetArrayFromImage(img_mask)

        if out_img.shape[0]-1 > 128 and out_img.shape[1]-1 > 128:
            patch_origin = randint(0, (((out_img.shape[0])-1)-128))
            patch_train = out_img[patch_origin:patch_origin+128, 
                                    patch_origin:patch_origin+128,
                                    patch_origin:patch_origin+128]
            patch_train = patch_train.astype('float32')
            patch_train /= 255.  # scale masks to [0, 1]
            patch_train = np.expand_dims(patch_train, axis = 3)  
            train_arrays_list.append(patch_train)
            print('train vol: '+str(image))
            patient_list_img.append(str(image))
            # sava patch of volume in numpy array.
            np.save('./patch_train_dataset/train_patch_'+str(image), patch_train)
            patch_mask = out_mask[patch_origin:patch_origin+128, 
                                     patch_origin:patch_origin+128,
                                     patch_origin:patch_origin+128]
            patch_mask = patch_mask.astype('float32')
            patch_mask /= 255.  # scale masks to [0, 1]
            patch_mask = np.expand_dims(patch_mask, axis = 3)
            mask_arrays_list.append(patch_mask)
            print('train mask: '+str(image))
            patient_list_mask.append(str(image))
            # sava patch of mask in numpy array.
            np.save('./patch_train_mask_dataset/mask_patch_'+str(image), patch_mask)
           
        else:
            print('Patient number '+str(image)+' has not enough slices or' 
                  'rows/columns for 128x128x128 patches')   
        
    
    print('Loading of train and mask datasets done.')
    print('Saving to .npy files done.')
    
    return train_arrays_list, mask_arrays_list, patient_list_img, patient_list_mask



# The next definitions are used to visualize the 3d volumes in Jupyter notebook.

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
                
def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

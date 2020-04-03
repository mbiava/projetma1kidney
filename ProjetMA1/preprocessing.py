#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:07:22 2020

@author: mathieubiava

inspired by: https://github.com/muellerdo/kits19.MIScnn.git
             https://github.com/imlab-uiip/lung-segmentation-2d.git
"""

import nibabel as nib
import numpy as np 
import math
import os

#-----------------------------------------------------#
#          Patches and Batches Preprocessing          #
#-----------------------------------------------------#

def preprocessing(cases) :
    
    images = np.array
    masks = np.array
    
    for i in cases :
        
        try:
            i = int(i)
            case_id = "case_{:05d}".format(i)
        except ValueError:
            case_id = i
        
        case_path = os.path.join("data", case_id)
        volume = nib.load(os.path.join(case_path, "imaging.nii.gz"))
        vol_data = volume.get_data()
        vol_data = np.reshape(vol_data, vol_data.shape + (1,))
        
        segmentation = nib.load(os.path.join(case_path, "segmentation.nii.gz"))
        seg_data = segmentation.get_data()
        seg_data = np.reshape(seg_data, seg_data.shape + (1,))

        mean = np.mean(vol_data)
        std = np.std(vol_data)
        
        vol_data = (vol_data - mean) / std

        patches_vol = slice_3Dmatrix(vol_data,
                                     (48, 128, 128),
                                     (12, 32, 32))
        
        
        patches_seg = slice_3Dmatrix(seg_data,
                                     (48, 128, 128),
                                     (12, 32, 32))
        
        for j in reversed(range(0, len(patches_seg))):
            patches_seg[j][patches_seg[j]==2] = 1 
            if np.sum(patches_seg[j]==1)<2500 : # not np.any(patches_seg[j] != 0) :
                del patches_vol[j]
                del patches_seg[j]

        batch_vol = np.concatenate(patches_vol[0:len(patches_vol)], axis=0)
        batch_seg = np.concatenate(patches_seg[0:len(patches_seg)], axis=0)
        
        if i == cases[0] :
            images = batch_vol
            masks = batch_seg
        else : 
            images = np.concatenate((images, batch_vol), axis=0)
            masks = np.concatenate((masks, batch_seg), axis=0)

    return images, masks

def preprocessing_predict(cases) :
    
    images = np.array
    
    for i in cases :
        
        try:
            i = int(i)
            case_id = "case_{:05d}".format(i)
        except ValueError:
            case_id = i
        
        case_path = os.path.join("data", case_id)
        volume = nib.load(os.path.join(case_path, "imaging.nii.gz"))
        vol_data = volume.get_data()
        vol_data = np.reshape(vol_data, vol_data.shape + (1,))
        
        mean = np.mean(vol_data)
        std = np.std(vol_data)
        
        vol_data = (vol_data - mean) / std

        patches_vol = slice_3Dmatrix(vol_data,
                                     (48, 128, 128),
                                     (12, 32, 32))

        batch_vol = np.concatenate(patches_vol[0:len(patches_vol)], axis=0)
       
        if i == cases[0] :
            images = batch_vol

        else : 
            images = np.concatenate((images, batch_vol), axis=0)

    return images

#-----------------------------------------------------#
#                 Slice 3D Matrices                   #
#-----------------------------------------------------#

def slice_3Dmatrix(array, window, overlap):
    
    steps_x = int(math.ceil((len(array) - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((len(array[0]) - overlap[1]) /
                            float(window[1] - overlap[1])))
    steps_z = int(math.ceil((len(array[0][0]) - overlap[2]) /
                            float(window[2] - overlap[2])))

    patches = []
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                # Define window edges
                x_start = x*window[0] - x*overlap[0]
                x_end = x_start + window[0]
                y_start = y*window[1] - y*overlap[1]
                y_end = y_start + window[1]
                z_start = z*window[2] - z*overlap[2]
                z_end = z_start + window[2]

                if(x_end > len(array)):
                    
                    x_start = len(array) - window[0]
                    x_end = len(array)
                    
                    if x_start < 0:
                        x_start = 0
                if(y_end > len(array[0])):
                    y_start = len(array[0]) - window[1]
                    y_end = len(array[0])
                if(z_end > len(array[0][0])):
                    z_start = len(array[0][0]) - window[2]
                    z_end = len(array[0][0])
                
                for i in range(x_start,x_end) :
                    window_cut = array[i,y_start:y_end,z_start:z_end]
                    window_slice = np.reshape(window_cut, (1,) + window_cut.shape)
                    patches.append(window_slice)
                
    return patches

#-----------------------------------------------------#
#              Concatenate 3D Matrices                #
#-----------------------------------------------------#

def concat_3Dmatrices(patches, image_size, window, overlap):

    steps_x = int(math.ceil((image_size[0] - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((image_size[1] - overlap[1]) /
                            float(window[1] - overlap[1])))
    steps_z = int(math.ceil((image_size[2] - overlap[2]) /
                            float(window[2] - overlap[2])))

    matrix_x = None
    matrix_y = None
    matrix_z = None
    pointer = 0
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                
                pointer = z + y*steps_z + x*steps_y*steps_z
                
                if z == 0:
                    matrix_z = patches[pointer]
                else:
                    matrix_p = patches[pointer]
                    
                    slice_overlap = calculate_overlap(z, steps_z, overlap,
                                                      image_size, window, 2)
                    matrix_z, matrix_p = handle_overlap(matrix_z, matrix_p,
                                                        slice_overlap,
                                                        axis=2)
                    matrix_z = np.concatenate((matrix_z, matrix_p),
                                              axis=2)
            
            if y == 0:
                matrix_y = matrix_z
            else:
                
                slice_overlap = calculate_overlap(y, steps_y, overlap,
                                                  image_size, window, 1)
                matrix_y, matrix_z = handle_overlap(matrix_y, matrix_z,
                                                    slice_overlap,
                                                    axis=1)
                matrix_y = np.concatenate((matrix_y, matrix_z), axis=1)
        
        if x == 0:
            matrix_x = matrix_y
        else:
            
            slice_overlap = calculate_overlap(x, steps_x, overlap,
                                              image_size, window, 0)
            matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y,
                                                slice_overlap,
                                                axis=0)
            matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
    
    return(matrix_x)

#-----------------------------------------------------#
#          Subroutines for the Concatenation          #
#-----------------------------------------------------#

def calculate_overlap(pointer, steps, overlap, image_size, window, axis):
            
            if pointer == steps-1 and not (image_size[axis]-overlap[axis]) \
                                            % (window[axis]-overlap[axis]) == 0:
                current_overlap = window[axis] - \
                                  (image_size[axis] - overlap[axis]) % \
                                  (window[axis] - overlap[axis])
            
            else:
                current_overlap = overlap[axis]
            
            return current_overlap


def handle_overlap(matrixA, matrixB, overlap, axis=0):
    
    idxA = [slice(None)] * matrixA.ndim
    idxA[axis] = range(len(matrixA)-overlap, len(matrixA))
    sliceA = matrixA[tuple(idxA)]
    
    idxB = [slice(None)] * matrixB.ndim
    idxB[axis] = range(0, overlap)
    sliceB = matrixB[tuple(idxB)]
    
    matrixA[tuple(idxA)] = np.mean((sliceA, sliceB), axis=0)
    
    matrixB = np.delete(matrixB, [range(0, overlap)], axis=axis)
    
    return matrixA, matrixB


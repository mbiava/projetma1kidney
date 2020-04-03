#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:07:22 2020

@author: mathieubiava

inspired by: https://github.com/muellerdo/kits19.MIScnn.git
             https://github.com/imlab-uiip/lung-segmentation-2d.git
"""

import os
import math
import numpy as np
import nibabel as nib

from keras.optimizers import Adam 

from keras.models import model_from_json

from modeltraining import dice_coefficient, dice_coefficient_loss

from preprocessing import preprocessing_predict,concat_3Dmatrices

from starter_code.visualize import visualize

#-----------------------------------------------------#
#                   Load the model                    #
#-----------------------------------------------------#

inpath_model = os.path.join("model",  "Unet2D.model.json")
inpath_weights = os.path.join("model", "Unet2D.weights.h5")

json_file = open(inpath_model, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(inpath_weights)

metrics = [dice_coefficient, 'accuracy']

model.compile(optimizer=Adam(lr=0.001),
                   loss=dice_coefficient_loss,
                   metrics=metrics)

#model.summary()

#-----------------------------------------------------#
#                   Model predict                     #
#-----------------------------------------------------#

cases = [0] # coded for only one case at the time

images = preprocessing_predict(cases)

pred_seg = model.predict(images,
                         verbose=1)

#-----------------------------------------------------#
#             Save prediction & Visualize             #
#-----------------------------------------------------#

try:
    cases[0] = int(cases[0])
    case_id = "case_{:05d}".format(cases[0])
except ValueError:
    case_id = cases[0]

case_path = os.path.join("data", case_id)

volume = nib.load(os.path.join(case_path, "imaging.nii.gz"))
vol_data = volume.get_data()

segmentation = nib.load(os.path.join(case_path, "segmentation.nii.gz"))
seg_data = segmentation.get_data()
seg_data[seg_data==2] = 1 


a = int(math.ceil(len(pred_seg)/48))

pred_seg = np.reshape(pred_seg,(a,48,128,128,1))

pred_seg = concat_3Dmatrices(pred_seg, image_size=vol_data.shape, window=(48, 128, 128),
                             overlap=(12, 32, 32) )


if not os.path.exists("predictions"):
        os.mkdir("predictions")
        
nifti = nib.Nifti1Image(pred_seg, None)

nib.save(nifti, os.path.join("predictions",
                             "prediction_" + str(cases[0]).zfill(5) + ".nii.gz"))
    

pred_seg = pred_seg[:,:,:,0]

pred_seg[pred_seg<=0.5] = 0

pred_seg[pred_seg>0.5] = 1

visualize(vol_data, pred_seg, "evaluation")

visualize(vol_data, seg_data, "evaluation")

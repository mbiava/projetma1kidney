#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:07:22 2020

@author: mathieubiava

inspired by: https://github.com/muellerdo/kits19.MIScnn.git
             https://github.com/imlab-uiip/lung-segmentation-2d.git
"""
import matplotlib.pyplot as plt
import random
import os

from keras.optimizers import Adam 
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from untitled0 import preprocessing

cases = list(range(0,210))

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def UNet2D(inp_shape, k_size=3):
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)
    data = Input(shape=inp_shape)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(pool3)
    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(pool4)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up1)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv6)
    merged1 = concatenate([conv4, conv6], axis=merge_axis)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged1)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up2)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv7)
    merged2 = concatenate([conv3, conv7], axis=merge_axis)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged2)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(up3)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv8)
    merged3 = concatenate([conv2, conv8], axis=merge_axis)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(merged3)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(up4)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv9)
    merged4 = concatenate([conv1, conv9], axis=merge_axis)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(merged4)

    conv10 = Convolution2D(filters=1, kernel_size=k_size, padding='same', activation='sigmoid')(conv9)

    output = conv10
    model = Model(data, output)
    return model

#-----------------------------------------------------#
#              Construction of the model              #
#-----------------------------------------------------#

metrics = [dice_coefficient, 'accuracy']

model = UNet2D(inp_shape=(128, 128, 1))

model.summary()

model.compile(optimizer=Adam(lr=0.001),
                      loss=dice_coefficient_loss,
                      metrics=metrics)

#-----------------------------------------------------#
#               Training of the model                 #
#-----------------------------------------------------#

model_checkpoint = ModelCheckpoint('weight.h5', monitor='val_loss', save_best_only=True)

cases = random.shuffle(cases)

images, masks = preprocessing(cases)

history = model.fit(images, masks, batch_size=10, 
                    epochs=15, verbose=1, shuffle=True,
                    validation_split=0.2,
                    callbacks=[model_checkpoint])

#-----------------------------------------------------#
#                    Save the model                   #
#-----------------------------------------------------#

if not os.path.exists("model"):
    os.mkdir("model")

outpath_model = os.path.join("model",
                                 "Unet2D.model.json")

outpath_weights = os.path.join("model",
                                 "Unet2D.weights.h5")

model_json = model.to_json()

with open(outpath_model, "w") as json_file:
    json_file.write(model_json)

model.save_weights(outpath_weights)

#-----------------------------------------------------#
#                    Plot the loss                    #
#-----------------------------------------------------#

eva_path = "evaluation"
prefix = "split_validation"

if not os.path.exists(eva_path):
    os.mkdir(eva_path)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Set', 'Validation Set'], loc='upper left')
out_path = os.path.join(eva_path,
                        "acc." + str(prefix) + ".png")
plt.savefig(out_path)
plt.close()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Dice Coefficient Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Set', 'Validation Set'], loc='upper left')
out_path = os.path.join(eva_path,
                        "dice_coefficient_loss." + str(prefix) + ".png")
plt.savefig(out_path)
plt.close()

print('end plot')

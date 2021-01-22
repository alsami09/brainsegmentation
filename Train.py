import os
import matplotlib.pyplot as plt
import silence_tensorflow.auto
import tensorflow as tf                                              
from tensorflow import keras
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from tensorflow.python.keras import losses
import tensorflow.keras.backend as K
import numpy as np
import glob
import cv2
from tensorflow.keras.utils import plot_model

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random

random.seed(9494)
tf.random.set_seed(9494)

import models 
from preprocessing import load_data,tf_dataset


Width = 128
Height = 128
Channel = 1
batch_size = 32
lr = 0.001
epochs = 250
nClasses = 2

#loading the Data
(train_x, train_y),(test_x, test_y) = load_data()
train_dataset = tf_dataset(train_x, train_y, batch=batch_size,nClasses=nClasses)
test_dataset = tf_dataset(test_x, test_y, batch=batch_size,nClasses=nClasses)

#Loss function
def soft_dice_loss(y_true, y_pred, epsilon=0.00001):
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    dice_numerator = 2. * K.sum(y_true * y_pred, axes) + epsilon
    dice_denominator = K.sum(y_true**2, axes) + K.sum(y_pred**2, axes) + epsilon
    dice_loss = 1 - K.mean((dice_numerator)/(dice_denominator))
    loss = dice_loss + losses.binary_crossentropy(y_true, y_pred)
    return loss

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou




#define and compile model
tumor_model = models.seg_model.Unet(Width,Height,Channel,nClasses)
opt = tf.keras.optimizers.Adam(lr)
# tumor_model.compile(optimizer=opt, loss=soft_dice_loss, metrics=[tf.keras.metrics.MeanIoU(num_classes=nClasses)])
tumor_model.compile(optimizer=opt, loss=soft_dice_loss, metrics=[iou_coef,'accuracy'])
tumor_model.summary()

train_steps = len(train_x)//batch_size
valid_steps = len(test_x)//batch_size

earlystopper = EarlyStopping(monitor='val_iou_coef',patience=20, verbose=1,mode='max')
checkpointer = ModelCheckpoint('model-tissue_wm.h5', verbose=1, save_best_only=True)


#train model
history = tumor_model.fit(train_dataset,
                    steps_per_epoch=train_steps,
                    validation_data=test_dataset,
                    validation_steps=valid_steps,
                    epochs=epochs, callbacks=[earlystopper,checkpointer])


#Plot training keys 

plt.subplot(1, 2, 1)
plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss WM')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig("model loss 31-12  tissue4", format = "png")

plt.subplot(1, 2, 2)
plt.figure(1)
plt.plot(history.history['iou_coef'])
plt.plot(history.history['val_iou_coef'])
plt.title('model mean IOU WM')
plt.ylabel('mean IOU')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model IOU 31-12 tissue4", format = "png")
plt.show()
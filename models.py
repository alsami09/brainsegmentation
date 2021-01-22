import os
import matplotlib.pyplot as plt
import tensorflow as tf                                              
from tensorflow import keras
from tensorflow.python.keras.backend import dropout
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

class seg_model():

    def Unet(Width,Height,Channel,nClasses):

        inputs = Input((Width,Height,Channel))

        convpars = dict(kernel_size=(3,3),activation='elu', padding='same',kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.001))

        c1 = Conv2D(16, **convpars) (inputs)
        c1 = Dropout(0.1) (c1)
        c1 = Conv2D(16,**convpars) (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(32, **convpars) (p1)
        c2 = Dropout(0.1) (c2)
        c2 = Conv2D(32, **convpars) (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(64, **convpars) (p2)
        c3 = Dropout(0.2) (c3)
        c3 = Conv2D(64, **convpars) (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(128, **convpars) (p3)
        c4 = Dropout(0.2) (c4)
        c4 = Conv2D(128, **convpars) (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(256, **convpars) (p4)
        c5 = Dropout(0.3) (c5)
        c5 = Conv2D(256, **convpars) (c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, **convpars) (u6)
        c6 = Dropout(0.2) (c6)
        c6 = Conv2D(128, **convpars) (c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, **convpars) (u7)
        c7 = Dropout(0.2) (c7)
        c7 = Conv2D(64, **convpars) (c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, **convpars) (u8)
        c8 = Dropout(0.1) (c8)
        c8 = Conv2D(32, **convpars) (c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, **convpars) (u9)
        c9 = Dropout(0.1) (c9)
        c9 = Conv2D(16, **convpars) (c9)

        outputs = Conv2D(nClasses, (1, 1), activation='sigmoid') (c9)
        model = Model(inputs=[inputs], outputs=[outputs])

        return model




    def Unet_dynamic(Width,Height,n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
        inputs = Input(shape=(Height, Width, in_channels))
        x = inputs
        
        convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
        
        #downstream
        skips = {}
        for level in range(n_levels):
            for _ in range(n_blocks):
                x = Conv2D(initial_features * 2 ** level, **convpars)(x)

            if level < n_levels - 1:
                skips[level] = x
                x = MaxPooling2D(pooling_size)(x)
                
        # upstream
        for level in reversed(range(n_levels-1)):
            x = Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
            x = concatenate()([x, skips[level]])
            for _ in range(n_blocks):
                x = Conv2D(initial_features * 2 ** level, **convpars)(x)      
        # output
        activation = 'sigmoid' if out_channels == 1 else 'softmax'
        x = Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
        model = Model(inputs=[inputs], outputs=[x])
        return model

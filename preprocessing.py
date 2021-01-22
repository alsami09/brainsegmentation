import os
import tensorflow as tf                                              
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import numpy as np
import glob
import cv2

import random


random.seed(9494)
tf.random.set_seed(9494)

# Define paths to data
data_dir = 'D:\Multiclass-Segmentation-in-Unet-master\Multiclass-Segmentation-in-Unet-master\slices_wm'
data_dir_train = os.path.join(data_dir, 'Training')
# The images should be stored under: "data/slices/training/img/img"
data_dir_train_image = os.path.join(data_dir_train, 'img/img\*')
# The images should be stored under: "data/slices/training/mask/img"
data_dir_train_mask = os.path.join(data_dir_train, 'mask/img\*')

data_dir_test = os.path.join(data_dir, 'Testing')
# The images should be stored under: "data/slices/test/img/img"
data_dir_test_image = os.path.join(data_dir_test, 'img/img\*')
# The images should be stored under: "data/slices/test/mask/img"
data_dir_test_mask = os.path.join(data_dir_test, 'mask/img\*')


Width = 128
Height = 128
Channel = 1
batch_size = 32
    
def load_data():

    train_x = glob.glob(data_dir_train_image)
    train_y = glob.glob(data_dir_train_mask)
    test_x = glob.glob(data_dir_test_image)
    test_y = glob.glob(data_dir_test_mask)

    train_image = []
    train_label = []
    test_image = []
    test_label = []

    for i , j in zip(train_x,train_y):
        x = np.sum(cv2.imread(j))
        if x != 0:
            train_image.append(i)
            train_label.append(j)
        else:
            print("x")

    for i , j in zip(test_x,test_y):
        x = np.sum(cv2.imread(j))
        if x != 0:
            test_image.append(i)
            test_label.append(j)
        else:
            print("x")

    print("Total Train Slices:", len(train_x),len(train_y), " ROI Train Slices:",len(train_image),len(train_label))
    print("Total Test Slices:", len(test_x),len(test_y), " ROI Test Slices:",len(test_image),len(test_label))

    return (train_image, train_label), (test_image, test_label)

def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = x / 255
    x = cv2.resize(x, (Width, Height))
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    return x

def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (Width, Height))
    x = x.astype(np.int32)
    return x


def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        image = read_image(x)
        mask = read_mask(y)

        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, nClass, dtype=tf.float32)
    # mask = mask[:,:,1:]
    image.set_shape([Width, Height, Channel])
    mask.set_shape([Width, Height, nClass])
    return image, mask


def tf_dataset(x, y, batch=8,nClasses=1):
    global nClass
    nClass = nClasses
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # dataset = dataset.shuffle(buffer_size=500000) 
    dataset = dataset.map(preprocess,num_parallel_calls = None)
    dataset = dataset.batch(batch, drop_remainder=True)
    # dataset = dataset.repeat()
    dataset = dataset.prefetch(2)

    return dataset








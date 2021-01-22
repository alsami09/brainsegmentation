import tensorflow as tf
from tensorflow.keras.models import load_model
import nibabel as nib
from niwidgets import NiftiWidget
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Define constants

HOUNSFIELD_MIN = 0
HOUNSFIELD_MAX = 65535
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

SLICE_X = True
SLICE_Y = True
SLICE_Z = True

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)


def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN)/HOUNSFIELD_RANGE


targetName = 'inferance'
targetImagePath = "C:\Users\manso\Desktop\MRI\IMAGE_TEST\SUB_42_T1fs_conform.nii.gz"
targetMaskPath  = "C:\Users\manso\Desktop\MRI\Mask_TEST_CSF\SUB42_MASK_CSF.nii.gz"

imgTargetNii = nib.load(targetImagePath)
imgMaskNii = nib.load(targetMaskPath)

imgTarget = normalizeImageIntensityRange(imgTargetNii.get_fdata())
imgMask = imgMaskNii.get_fdata()


with tf.device('/cpu:0'):
    model = load_model('UNET-MRI_CSF_Segmentation_128_128.h5')

def scaleImg(img, height, width):
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

sliceIndex = 150

#Single slicing prediction
plt.figure(figsize=(15,15))
imgSlice = imgTarget[:,sliceIndex,:]
imgDimX, imgDimY = imgSlice.shape
imgSliceScaled = scaleImg(imgSlice, IMAGE_HEIGHT, IMAGE_WIDTH)
plt.subplot(1,2,1)
plt.imshow(imgSlice, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(imgSliceScaled, cmap='gray')
plt.show()
imgSlice.shape, imgSliceScaled.shape


# show input mask slice
plt.figure(figsize=(15,15))
maskSlice = imgMask[:,sliceIndex,:]
maskSliceScaled = scaleImg(maskSlice, IMAGE_HEIGHT, IMAGE_WIDTH)
plt.subplot(1,2,1)
plt.imshow(maskSlice, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(maskSliceScaled, cmap='gray')
plt.show()
maskSlice.shape, maskSliceScaled.shape


with tf.device('/cpu:0'):
# Predict with UNET model
    plt.figure(figsize=(15,15))
    imageInput = imgSliceScaled[np.newaxis,:,:,np.newaxis]
    maskPredict = model.predict(imageInput)[0,:,:,0]
    maskPredictScaled = scaleImg(maskPredict, imgDimX, imgDimY)
    plt.subplot(1,2,2)
    plt.imshow(maskPredict, cmap='gray')
    plt.subplot(1,2,1)
    plt.imshow(maskPredictScaled, cmap='gray')
    plt.show()
maskPredictScaled.shape, maskPredict.shape


def predictVolume(inImg, toBin=True):
    (xMax, yMax, zMax) = inImg.shape
    
    outImgX = np.zeros((xMax, yMax, zMax))
    outImgY = np.zeros((xMax, yMax, zMax))
    outImgZ = np.zeros((xMax, yMax, zMax))
    
    cnt = 0.0
    if SLICE_X:
        cnt += 1.0
        for i in range(xMax):
            img = scaleImg(inImg[i,:,:], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            outImgX[i,:,:] = scaleImg(tmp, yMax, zMax)
    if SLICE_Y:
        cnt += 1.0
        for i in range(yMax):
            img = scaleImg(inImg[:,i,:], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            outImgY[:,i,:] = scaleImg(tmp, xMax, zMax)
    if SLICE_Z:
        cnt += 1.0
        for i in range(zMax):
            img = scaleImg(inImg[:,:,i], IMAGE_HEIGHT, IMAGE_WIDTH)[np.newaxis,:,:,np.newaxis]
            tmp = model.predict(img)[0,:,:,0]
            outImgZ[:,:,i] = scaleImg(tmp, xMax, yMax)
            
    outImg = (outImgX + outImgY + outImgZ)/cnt
    if(toBin):
        outImg[outImg>0.9] = 1.0
        outImg[outImg<=0.9] = 0.0
    return outImg


with tf.device('/cpu:0'):
    predImg = predictVolume(imgTarget)


my_widget = NiftiWidget(imgTargetNii)
my_widget.nifti_plotter(colormap='gray')

my_widget = NiftiWidget(nib.dataobj_images.DataobjImage(predImg))
my_widget.nifti_plotter(colormap='gray')

my_widget = NiftiWidget(imgMaskNii)
my_widget.nifti_plotter(colormap='gray')


from skimage.measure import marching_cubes_lewiner

vertices,faces,_,_ = marching_cubes_lewiner(predImg)

import meshplot as mp

mp.plot(vertices, faces, return_plot=False)

from stl import mesh

def dataToMesh(vert, faces):
    mm = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mm.vectors[i][j] = vert[f[j],:]
    return mm


mm = dataToMesh(vertices, faces)
mm.save('head-segmented.stl')
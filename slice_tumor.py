import os, glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


# Paths train data 

# Raw train data
traindataInputPath = 'D:\Multiclass-Segmentation-in-Unet-master\Multiclass-Segmentation-in-Unet-master\Dataset\Train'
trainimagePathInput = os.path.join(traindataInputPath, 'IMAGE/')
trainmaskPathInput = os.path.join(traindataInputPath, 'MASK/')

#post processed / sliced train data
traindataOutputPath = 'D:\Multiclass-Segmentation-in-Unet-master\Multiclass-Segmentation-in-Unet-master\slices_Tumor\Training'
trainimageSliceOutput = os.path.join(traindataOutputPath, 'img/img/')
trainmaskSliceOutput = os.path.join(traindataOutputPath, 'mask/img/')

# Paths to test data 
# Raw test data
testdataInputPath = 'D:\Multiclass-Segmentation-in-Unet-master\Multiclass-Segmentation-in-Unet-master\Dataset\Test'
testimagePathInput = os.path.join(testdataInputPath, 'IMAGE/')
testmaskPathInput = os.path.join(testdataInputPath, 'MASK/')

#post process / sliced test data
testdataOutputPath = 'D:\Multiclass-Segmentation-in-Unet-master\Multiclass-Segmentation-in-Unet-master\slices_Tumor\Testing'
testimageSliceOutput = os.path.join(testdataOutputPath, 'img/img/')
testmaskSliceOutput = os.path.join(testdataOutputPath, 'mask/img/')

# Constants

# Image normalization
HOUNSFIELD_MIN = 0
HOUNSFIELD_MAX = 4500
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

# Slicing and saving
SLICE_X = True
SLICE_Y = True
SLICE_Z = True
SLICE_DECIMATE_IDENTIFIER = 3


#test the train data images
trainimgPath = os.path.join(trainimagePathInput, 'BRATS_001.nii.gz')
img = nib.load(trainimgPath).get_fdata()
print(np.min(img)), print(np.max(img)), print(img.shape), print(type(img))

#test the train data masks
trainmaskPath = os.path.join(trainmaskPathInput, 'BRATS_001.nii.gz')
mask = nib.load(trainmaskPath).get_fdata()
print(np.min(mask)), print(np.max(mask)), print(mask.shape), print(type(mask))

#test the test data images
testimgPath = os.path.join(testimagePathInput, 'BRATS_004.nii.gz')
img = nib.load(testimgPath).get_fdata()
print(np.min(img)), print(np.max(img)), print(img.shape), print(type(img))

#test the test data masks
testmaskPath = os.path.join(testmaskPathInput, 'BRATS_004.nii.gz')
mask = nib.load(testmaskPath).get_fdata()
print(np.min(mask)), print(np.max(mask)), print(mask.shape), print(type(mask))


# Normalize image
def normalizeImageIntensityRange(img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

# Read image or mask volume
def readImageVolume(imgPath, normalize=False):
    img = nib.load(imgPath).get_fdata()
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        return img

# Save volume slice to file
def saveSliceImg(img, fname, path):
    img = np.uint8(img * 255)
    # img = resize(img)
    fout = os.path.join(path, f'{fname}.png')
    cv2.imwrite(fout,img)
    print(f'[+] Slice saved: {fout}', end='\r')

def saveSliceMask(img, fname, path):
    img = np.uint8(img)
    # img = resize(img)
    fout = os.path.join(path, f'{fname}.png')
    cv2.imwrite(fout,img)
    print(f'[+] Slice saved: {fout}', end='\r')


    # Slice image in all directions and save
def sliceAndSaveVolumeImage(vol, fname, path):
    (dimx, dimy, dimz,dimq) = vol.shape
    print(dimx, dimy, dimz,dimq)
    cnt = 0
    if SLICE_X:
        cnt += dimx
        print('Slicing X: ')
        for i in range(dimx):
            saveSliceImg(vol[i,:,:,1], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)
            
    if SLICE_Y:
        cnt += dimy
        print('Slicing Y: ')
        for i in range(dimy):
            saveSliceImg(vol[:,i,:,1], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)
            
    if SLICE_Z:
        cnt += dimz
        print('Slicing Z: ')
        for i in range(dimz):
            saveSliceImg(vol[:,:,i,1], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)

    return cnt


# Slice Masks in all directions and save
def sliceAndSaveVolumeMask(vol, fname, path):
    (dimx, dimy, dimz) = vol.shape
    print(dimx, dimy, dimz)
    cnt = 0
    if SLICE_X:
        cnt += dimx
        print('Slicing X: ')
        for i in range(dimx):
            saveSliceMask(vol[i,:,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)
            
    if SLICE_Y:
        cnt += dimy
        print('Slicing Y: ')
        for i in range(dimy):
            saveSliceMask(vol[:,i,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)
            
    if SLICE_Z:
        cnt += dimz
        print('Slicing Z: ')
        for i in range(dimz):
            saveSliceMask(vol[:,:,i], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)

    return cnt

    # Read and process train image volumes
for index, filename in enumerate(sorted(glob.iglob(trainimagePathInput+'*.nii.gz'))):
    
    img = readImageVolume(filename, True)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    time.sleep(6)
    numOfSlices = sliceAndSaveVolumeImage(img, 'MRI'+str(index), trainimageSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')

#     # Read and process test image volumes
for index, filename in enumerate(sorted(glob.iglob(testimagePathInput+'*.nii.gz'))):
      
    img = readImageVolume(filename, True)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    time.sleep(6)
    numOfSlices = sliceAndSaveVolumeImage(img, 'MRI'+str(index), testimageSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')


    # Read and process train image mask volumes
for index, filename in enumerate(sorted(glob.iglob(trainmaskPathInput+'*.nii.gz'))):
    
    img = readImageVolume(filename, False)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    time.sleep(6)
    numOfSlices = sliceAndSaveVolumeMask(img, 'MRI'+str(index), trainmaskSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')

    # Read and process test image mask volumes
for index, filename in enumerate(sorted(glob.iglob(testmaskPathInput+'*.nii.gz'))):
    
    img = readImageVolume(filename, False)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    time.sleep(6)
    numOfSlices = sliceAndSaveVolumeMask(img, 'MRI'+str(index), testmaskSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')
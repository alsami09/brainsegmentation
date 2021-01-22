import numpy as np
import logging
import nibabel as nib
import os
from scipy.ndimage import map_coordinates


logger = logging.getLogger("py.warnings")
width = 72


def prepare_vols(fin, out_format="keep", vx_size=None, img_dims=None,
                 center=False, write_vols=True):
    """Transform a set of volumes to LAS (radiological) space. Optionally,
    resample the image to a new set of voxel sizes and image dimensions. 
    Assumes that all input volumes have the same voxel size and image 
    dimensions.
    
    PARAMETERS
    ----------
    fin : list
        List of strings (filenames) or nibabel image objects.
    out_format : str
        How to write the new volumes. Options are "standard", "keep",
        "custom", "no-conform") (default = "keep").
    vx_size : array_like
        Array specifying voxel sizes (only applies if out_format = "custom").
    img_dims : array_like
        Array specifying image dimensions (only applies if out_format = 
        "custom").
    center : bool, optional
        Whether or not to center the data. The center is assumed to be
            M^(-1)*[0,0,0,1]
        where M is the affine transformation matrix between voxel coordinates
        (indices) and physical or standard space (default = False).
    write_vols : bool, optional
        Write conforming volumes to disk (default = True).
    
    RETURNS
    ----------
    vols_conform : list
        List of nibabel image objects, i.e. the volumes in the conformed space. 
    """     
    # Check inputs
    out_format = out_format.lower()
    
    if type(fin) == str:
        fin=[fin]
        
    # if input is not iterable, make it so
    try:
        next(iter(fin))
    except TypeError:
        fin = [fin]
    
    if vx_size is None:
        vx_size = []
    if img_dims is None:
        img_dims = []
        
    vols=[]
    for f in fin:
        if type(f) == str:
            try:
                vols.append(nib.load(f))
            except:
                raise IOError(f+" does not exist!") 
        elif type(f) in [nib.nifti1.Nifti1Image,nib.nifti2.Nifti2Image,nib.nifti1.Nifti1Pair,nib.nifti2.Nifti2Pair]:
            vols.append(f)
        else:
            raise ValueError("Input volumes must be specified as either a string (path to volume) or a nibabel image object.")
    
    vols_conform = []

    for v in vols:
        hdr_orig = v.header
        fname_out = add2filename(v.get_filename(), "_conform_new2")
        fname_in = v.get_filename()
        
        if out_format == "no-conform":
            # If the files are already in the appropriate space simply copy and
            # rename the files
            log("Copying {0} to {1}", 
                [os.path.basename(fname_in),
                 os.path.basename(fname_out)])
                 
            data = v.get_data()
            M = v.affine            
        else:
            # Reorient to LAS (radiological) space
            log("Preparing {0} from {1}", 
                [os.path.basename(fname_out),
                 os.path.basename(fname_in)])
            log("-"*width)
            
            v = reorient2LAS(v)
            
            vx_size_original = get_vx_size(v)
            
            if out_format == "keep":
                vx_size  = vx_size_original
                img_dims = np.array(v.shape) 
            elif out_format == "standard":
                vx_size  = np.array([1]*3)
                img_dims = np.array([256]*3)
            elif (out_format == "custom") & (len(vx_size) == 3) & (len(img_dims) == 3):
                vx_size = np.array(vx_size)
                img_dims = np.array(img_dims)
            elif (out_format == "custom") & (len(vx_size) == 3):
                vx_size  = np.array(vx_size).astype(float)
                img_dims = np.array(v.shape)*vx_size_original/vx_size
            elif (out_format == "custom") & (len(img_dims) == 3):
                img_dims = np.array(img_dims)
                vx_size  = np.round(vx_size_original*np.array(v.shape) / img_dims.astype(np.float))
            else:
                raise KeyError("Unrecognized output format.")
            
            img_dims_int = np.ceil(img_dims).astype(np.int)
            
            log("Output format")
            print("")
            log("{:19s} {:>30s}",("Orientation","Left-Anterior-Superior (LAS)"))
            log("{:33s} [{:3.2f} {:3.2f} {:3.2f}]",("Voxel size", vx_size[0],vx_size[1],vx_size[2]))
            log("{:33s} [{:4d} {:4d} {:4d}]",("Image dimensions", img_dims_int[0],img_dims_int[1],img_dims_int[2]))
            print("")
            
            # The affine transformation matrix of the output image
            M = np.array([[-vx_size[0],         0 ,         0 ,  vx_size[0]*img_dims_int[0]/2.],
                          [         0 , vx_size[1],         0 , -vx_size[1]*img_dims_int[1]/2.],
                          [         0 ,         0 , vx_size[2],  vx_size[2]*(-img_dims_int[2]/2.+1)],
                          [         0 ,         0 ,         0 ,  1]])
            
            # Calculate the scaling factor between the new and the original
            # voxel size in each dimension to determine how densely to sample
            # the original space.
            sampling_density = np.abs( vx_size/vx_size_original.astype(np.float) )

            # Calculate the amount (in mm) by which to offset the new grid so
            # as to center it on the original data (floor it to minimize
            # smoothing effects at the expense of sampling a little
            # incorrectly).
            FOV_offset = np.round((np.array(v.shape)*vx_size_original-img_dims*vx_size)*sampling_density / 2. )

            if np.any(FOV_offset-1e-3 > 0) or np.any(img_dims*sampling_density + FOV_offset+1e-3 < v.shape):
                log("WARNING: Sampling the original volume with the current voxel size/image dimensions will reduce FOV!", level=logging.WARNING)
 
            img = v.get_data().copy()
            img_affine = v.affine.copy()
            
            if center:
                # Move to the center of the array the point corresponding to 
                # (0,0,0) in mm space (standard space or scanner space)
                t2c = np.round( (np.array(v.shape)-1)/2. - np.linalg.inv(img_affine).dot(np.array([0,0,0,1]) )[:3]).astype(np.int)              
                if np.any(t2c):
                    for i in range(3):
                        img = np.roll(img,t2c[i],axis=i)

            # Sample the new grid        
            # get grid coordinates (in mm) with the new voxel size
            grid = np.array(np.meshgrid(*tuple(map(np.arange,[0,0,0],img_dims*sampling_density,sampling_density)),indexing="ij")).reshape((3,-1))
            
            # final FOV to sample
            grid += np.array(FOV_offset)[:,np.newaxis]

            # Interpolate values of new coordinates by sampling the original
            # data matrix. Use mode="nearest" at edges to prevent coordinates JUST
            # outside from automatically being assigned a value of zero.
            # Zero pad to avoid filling of entire rows/columns
            if np.mean(sampling_density) < 1.05 and np.mean(sampling_density) > 0.95:
                spline_order = 0
            else:
                spline_order = 5
            
            data = map_coordinates( np.pad(img,1,mode="constant",constant_values=0) , grid+1 ,order=spline_order,mode="nearest").reshape(img_dims_int)
            data[data < 0] = 0
        
        # ensure that data is 16 bit unsigned integers and rescale to range
        data = (data / float(data.max()) * (2**16-1)).astype(np.uint16)
        
        # save image using the new affine transformation matrix, M
        vout = nib.Nifti1Image(data,M)
        vout.set_qform(M)
        vout.header.set_xyzt_units(*hdr_orig.get_xyzt_units())
        vout.set_filename(fname_out)
        
        vols_conform.append(vout)
        
        if write_vols:
            nib.save(vout,vout.get_filename())
         
        return vols_conform , vout


def add2filename(filename, text, where="end", useext=None):
    """Add text to filename, either at the front or end of the basename.
    Everything after the first punctuation mark is considered file extension.
    """
    
    where = where.lower()
    assert where in ["end","front"]
     
    directory = os.path.dirname(filename)
    basename = os.path.basename(filename).split(".")[0]
    ext = ".".join(os.path.basename(filename).split(".")[1:])

    if useext:
        ext = useext
        
    if where == "end":
        return os.path.join(directory,basename+text+"."+ext)
    elif where == "front":
        return os.path.join(directory,text+basename+"."+ext)


def reorient2LAS(v):
    """Reorients axes so as to bring the volume in to LAS (left-anterior-
    superior) space, i.e. radialogical convention. This reorientation is not
    robust, however, in that it relies on finding the primary direction of each
    axis and assigning it to the corresponding dimension. As such, if an axis
    is more than 45 degrees off, this will fail to correctly match an array
    axis with its corresponding dimension.
    
    PARAMETERS
    ----------
    v : nibabel image object
        The image volume.
        
    RETURNS
    -------  
    vout : nibabel image object
        reoriented nibabel image object in LAS (radiological) convention.
    """    
    qform_inv = np.linalg.inv(v.get_qform())[:3,:3]
    qform_inv_sign = np.sign(qform_inv)
    
    # Get sorting to LAS. This should work except for a few "pathological"
    # cases (e.g. two 45 deg axes)
    _, LASdim = np.where((np.abs(qform_inv) == np.max(np.abs(qform_inv),axis=0)).T)
    LASsign = qform_inv_sign[LASdim,range(3)]
    LASsign[0] *= -1

    # Apply to qform
    p = np.zeros_like(v.get_qform())
    p[-1,-1] = 1
    p[LASdim,range(3)] = 1
    
    dims = np.array(v.shape)
    dims = dims[LASdim] # dimensions of the reoriented image 

    # flip axes to LAS   
    flip = np.eye(4)
    for i in range(3):
        if LASsign[i] < 0:
            flip[i,i] = -1
            flip[i,-1] = dims[i]-1
    
    # Apply to data
    data = np.transpose(v.get_data(),LASdim) # permute axes
    if LASsign[0]<0: data = data[::-1,...]   # and flip...
    if LASsign[1]<0: data = data[:,::-1,:]
    if LASsign[2]<0: data = data[...,::-1]
    
    # Make reoriented image
    vout = nib.Nifti1Image(data,affine=(v.get_qform().dot(p)).dot(flip))
    vout.set_qform((v.get_qform().dot(p)).dot(flip))
    vout.header.set_xyzt_units(*v.header.get_xyzt_units())
    
    return vout


def get_vx_size(vol):
    """Return the voxel size of a volume.
    
    PARAMETERS
    ----------
    vol : nibabel image object
        Image volume.
    
    RETURNS
    ----------
    vx_size : numpy.ndarray
        Voxel size in all three dimensions.
    """    
    try:
        vx_size = np.linalg.norm(vol.affine[:3,:3], ord=2, axis=0)
    except AttributeError:
        raise IOError("Input must be a nibabel image object.")
        
    return vx_size


def log(msg, args=None, level=logging.INFO, width=72):
    """Log message with formatting.
    
    PARAMETERS
    ----------
    msg : str
        The message to log.
    args : list or str, optional
        List of arguments. Arguments are applied using .format, hence supplying
        arguments, msg should contain proper placeholders for these arguments
        (e.g., {0}, {1}, {:d} or with options such as {:06.2f}) (default = [],
        i.e. no arguments).        
    width : int, optional
        Wrapping width of msg (default = 72)
        GUILHERME: deprecated this argument, it only makes the log more confusing
    level : logging level, optional
        Specify logging level (default = logging.INFO)
        
    RETURNS
    ----------
    The message logged on logger named "logger".    
    """
    if args is None:
        args = []
    if type(args) not in [list,tuple]:
        args = [args]
        
    try:
        logger.log(level, msg.format(*args))
    # If there is an error in the message formating, logs the raw message
    except ValueError:
        logger.log(level, msg)    



# targetImagePath = r"D:\Git\MRI_Segmentation\ADNI_941_S_1311_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20080313131733540_S40710_I97341.nii"
# targetMaskPath  = r"D:\Multiclass-Segmentation-in-Unet-master\Multiclass-Segmentation-in-Unet-master\Dataset2\Test\MASK_CSF\SUB64_csf.nii.gz"

# imgTargetNii = nib.load(targetImagePath)
# # print("before",np.max(imgTargetNii),imgTargetNii.shape)
# x , imgTargetNii = prepare_vols(targetImagePath,out_format="keep")
# print("before",np.max(imgTargetNii),imgTargetNii.shape)

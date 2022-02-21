import nibabel as nib
import numpy as np
import torch

def getFramesFromNifty14D(inputFilePath, outputFilePath):
    img = nib.load(inputFilePath)
    affine = img.affine
    hdr = img.header
    hdr['dim'][0] = 3
    hdr['dim'][4] = 1
    data = img.get_fdata()

    frames = {}

    for i in range(img.shape[3]):
        frames['frame_'+str(i)] = data[:,:,:,i]

    for k,v in frames.items():
        temp = nib.Nifti1Image(v,affine=affine,header=hdr)
        nib.save(temp,outputFilePath + k + ".nii")

def normalizeImage(img: np.ndarray):
    '''
    :param img: A numpy array of un-normalized values of any range.
    :return: The same image, normalized to the range [0.0,1.0]
    '''
    max = np.max(img)
    min = np.min(img)
    temp = np.subtract(img, min)
    temp = np.divide(temp, (max - min))
    return temp

def _metricNCC(moving,target, windowWidth: int=9):
    ndims = len(moving.shape) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    window = [windowWidth] * ndims
    windowSize = np.prod(window)
    sumFilter = torch.ones([1,1,*window],dtype=torch.float32).cuda()
    padNo = window[0]//2
    if ndims == 1:
        stride = (1)
        padding = (padNo)
    elif ndims == 2:
        stride = (1, 1)
        padding = (padNo, padNo)
    else:
        stride = (1, 1, 1)
        padding = (padNo, padNo, padNo)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)
    m2 = torch.mul(moving,moving)
    t2 = torch.mul(target,target)
    mt = torch.mul(moving,target)

    sum_M = conv_fn(moving,sumFilter,stride=stride,padding=padding)
    sum_T = conv_fn(target,sumFilter,stride=stride,padding=padding)
    sum_M2 = conv_fn(m2,sumFilter,stride=stride,padding=padding)
    sum_T2 = conv_fn(t2,sumFilter,stride=stride,padding=padding)
    sum_MT = conv_fn(mt,sumFilter,stride=stride,padding=padding)

    norm_M = sum_M / windowSize
    norm_T = sum_T / windowSize

    cross_coef = sum_MT - (norm_T * sum_M) - (norm_M * sum_T) + (norm_M * norm_T * windowSize)
    var_M = sum_M2 - (2*norm_M*sum_M) + (norm_M*norm_M*windowSize)
    var_T = sum_T2 - (2*norm_T*sum_T) + (norm_T*norm_T*windowSize)

    cc = cross_coef * cross_coef / (var_M * var_T + 1e-5)
    return torch.mean(cc)

def _metricMSE(moving, target):
    se = torch.subtract(moving,target)
    se = torch.pow(se,2.0)
    return torch.mean(se)

def _augmentDimensions(imageDimensions: tuple, augmentation):
    temp = list(imageDimensions)
    if type(augmentation) == int:
        temp.append(augmentation)
    elif type(augmentation) == list:
        temp = temp + augmentation
    else:
        aug = list(augmentation)
        temp = temp + aug
    return tuple(temp)
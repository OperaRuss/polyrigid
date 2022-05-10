import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def getFramesFromNifty14D(inputFilePath, outputFilePath):
    '''
    Utility function to slice a 4D NIfTI-1 file into k 3D frames, and adjusts header files appropriately.
    :param inputFilePath: File path to a 4D NIfTI-1 file.
    :param outputFilePath: File path to output folder for all sequence frames.
    :return:
    '''
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
    Min-max image normaliztion.
    :param img: A numpy array of un-normalized values of any range.
    :return: The same image, normalized to the range [0.0,1.0]
    '''
    max = np.max(img)
    min = np.min(img)
    temp = np.subtract(img, min)
    if max == 0 and min == 0:
        return img
    else:
        temp = np.divide(temp, (max - min))
        return temp

def _getMetricNCC(moving,target, windowWidth: int=9):
    '''
    [DRAFT FUNCTION] Gives the Normalized Cross Correlation loss metric.  Implementation emulates Voxel Morph.
    :param moving: Torch tensor of the moving image.
    :param target: Torch tensor of the target image.
    :param windowWidth: Int value scalar for the window width. Default is 9 voxels. Should be odd.
    :return:
    '''
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

def _augmentDimensions(imageDimensions: tuple, augmentation):
    '''
    Helper function to concatenate a tensor shape with an augmented dimension. Not used in code at this time.
    :param imageDimensions: Tensor shape tuple.
    :param augmentation: augmentation desired for the tensor.  Ie: (1,2,3) [4,4] => (1,2,3,4,4)
    :return:
    '''
    temp = list(imageDimensions)
    if type(augmentation) == int:
        temp.append(augmentation)
    elif type(augmentation) == list:
        temp = temp + augmentation
    else:
        aug = list(augmentation)
        temp = temp + aug
    return tuple(temp)

def pltImage(tImg: torch.tensor, title: str="", cmap:str='gray',
             toShow: bool=True, toSave: bool=False,
             outPath:str="../images/results/", outFile: str="img.png"):
    '''
    Basic plotting helper.
    '''
    plt.imshow(tImg,cmap=cmap)
    plt.title(title)
    plt.axis('off')
    if(toSave):
        plt.savefig(outPath + outFile,bbox_inches='tight')
    if(toShow):
        plt.show()
    plt.close()

def jacobian_determinant_3d(displacementField):
    '''
    Implementation from
    https://github.com/cwmok/LapIRN/blob/bc45fe07ae289985e4de99e850b0257524e3132d/Code/miccai2020_model_stage.py#L781
    :param tDisplacementField: A displacement vector field passed as a PyTorch tensor.
    :return: Torch tensor with scalar jacobian determinants (non-log)
    '''
    dy = displacementField[:, 1:, :-1, :-1, :] - displacementField[:, :-1, :-1, :-1, :]
    dx = displacementField[:, :-1, 1:, :-1, :] - displacementField[:, :-1, :-1, :-1, :]
    dz = displacementField[:, :-1, :-1, 1:, :] - displacementField[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 0] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet
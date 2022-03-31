import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import matplotlib.pyplot as plt

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
    if max == 0 and min == 0:
        return img
    else:
        temp = np.divide(temp, (max - min))
        return temp

def _getMetricNCC(moving,target, windowWidth: int=9):
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

def _getMetricMSE(moving, target):
    se = torch.subtract(target,moving)
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

def _getDistanceToCompoonentRegion(componentSegmentation: np.ndarray):
    '''
    :param componentSegmentation: Binary image with foreground values as objects and all else as background.
    :return: Returns the exact Euclidean distance from a background pixel to the nearest foreground pixel.
    '''
    maxIntensity = np.max(componentSegmentation)
    invertedImage = np.subtract(maxIntensity, componentSegmentation)
    return ndimage.distance_transform_edt(invertedImage)

def _getRegionWeight(componentSegmentation: np.ndarray, gamma: float):
    '''
    :param componentSegmentation: Binary label image of a single component from the image.
    :param gamma: The relative weight assigned to this component.
    :return: Returns an imgae containing a diffusion of influence over the image space relative to the object.
    '''
    return (1.0 / (1.0 + (gamma * pow(_getDistanceToCompoonentRegion(componentSegmentation), 2))))


def _getWeightCommowick(componentSegmentations: dict, ratesOfDecay: dict):
    '''
    :param componentSegmentations: Dictionary of binary images for component regions
    :param ratesOfDecay: Dictionary of {label:weight} pairs where label = component segmentation label
    :return: Dictionary of normalized weight images summing to 1.0 at each voxel
    '''
    vCommowickWeights = {}
    for label, segmentation in componentSegmentations.items():
        vCommowickWeights[label] = _getRegionWeight(segmentation, ratesOfDecay[label])
    vSumImage = np.zeros(componentSegmentations[0].shape,dtype=np.float32)
    for image in vCommowickWeights.values():
        vSumImage += image
    vNormalizedWeights = {}
    for label, image in vCommowickWeights.items():
        vNormalizedWeights[label] = np.divide(image, vSumImage)
    return vNormalizedWeights

def rotX(radians: float,isTorch: bool=False):
    temp = np.array([[1,0,0,0],
                         [0,np.cos(radians),np.sin(radians),0],
                         [0,-np.sin(radians),np.cos(radians),0],
                         [0,0,0,1]],dtype=np.float32)
    if (isTorch):
        return torch.tensor(temp,dtype=torch.float32).cuda()
    else:
        return temp

def rotY(radians:float, isTorch: bool=False):
    temp = np.array([[np.cos(radians),0,-np.sin(radians),0],
                         [0,1,0,0],
                         [-np.sin(radians),0,np.cos(radians),0],
                         [0,0,0,1]],dtype=np.float32)
    if (isTorch):
        return torch.tensor(temp,dtype=torch.float32).cuda()
    else:
        return temp

def rotZ(radians: float, isTorch: bool=False):
    temp = np.array([[np.cos(radians),-np.sin(radians),0,0],
                         [np.sin(radians),np.cos(radians),0,0],
                         [0,0,1,0],
                         [0,0,0,1]],dtype=np.float32)
    if (isTorch):
        return torch.tensor(temp,dtype=torch.float32).cuda()
    else:
        return temp

def jacobian_determinant_3d(tDisplacementField):
    '''
    Implementation from
    https://github.com/cwmok/LapIRN/blob/bc45fe07ae289985e4de99e850b0257524e3132d/Code/miccai2020_model_stage.py#L781
    :param tDisplacementField:
    :return:
    '''
    dy = tDisplacementField[:,1:,:-1,:-1,:] - tDisplacementField[:,:-1,:-1,:-1,:]
    dx = tDisplacementField[:,:-1,1:,:-1,:] - tDisplacementField[:,:-1,:-1,:-1,:]
    dz = tDisplacementField[:,:-1,:-1,1:,:] - tDisplacementField[:,:-1,:-1,:-1,:]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1]*dz[:,:,:,:,2] - dy[:,:,:,:,2]*dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0]*dz[:,:,:,:,2] - dy[:,:,:,:,2]*dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0]*dz[:,:,:,:,0] - dy[:,:,:,:,1]*dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet

def _loss_JDet(tDisplacementField):
    neg_Jdet = -1.0 * jacobian_determinant_3d(tDisplacementField)
    selected_neg_Jdet = F.relu(neg_Jdet)
    return torch.mean(selected_neg_Jdet)

def _loss_Smooth(tDisplacementField):
    dy = torch.abs(tDisplacementField[:,1:,:,:,:] - tDisplacementField[:,:-1,:,:,:])
    dx = torch.abs(tDisplacementField[:,:,1:,:,:] - tDisplacementField[:,:,:-1,:,:])
    dz = torch.abs(tDisplacementField[:,:,:,1:,:] - tDisplacementField[:,:,:,:-1,:])
    return (torch.mean(dx*dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0

def pltImage(tImg: torch.tensor, title: str="", cmap:str='gray',
             toShow: bool=True, toSave: bool=False,
             outPath:str="../images/results/", outFile: str="img.png"):
    plt.imshow(tImg,cmap=cmap)
    plt.title(title)
    plt.axis('off')
    if(toSave):
        plt.savefig(outPath + outFile,bbox_inches='tight')
    if(toShow):
        plt.show()
    plt.close()
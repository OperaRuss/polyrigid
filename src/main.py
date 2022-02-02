''''
The goal of this repository is to give a basic implementation of the Log-Euclidean Polyrigid Image
Registration algorithm as outlined in Arsigny, et al., in the following article:

Vincent Arsigny, Olivier Commowick, Nicholas Ayache, Xavier Pennec. A Fast and Log-Euclidean
Polyaﬀine Framework for Locally Linear Registration. Journal of Mathematical Imaging and Vision,
Springer Verlag, 2009, 33 (2), pp.222-238. 10.1007/s10851-008-0135-9. inria-00616084

This implementation is part of Russell Wustenberg's work with the Visualization, Imaging and Data
Analysis (VIDA) research lab at New York University's Tandon School of Engineering.
'''

# External Modules
import SimpleITK as sitk
import numpy as np
import torch
from pytorch3d import transforms

# Custom Classes
import utilities as utils
import Weights
import torchfields


# STEP 1: Read in data
movingData = sitk.ReadImage("../images/moving.nii")
movingData = sitk.GetArrayFromImage(movingData)

fixedData = sitk.ReadImage("../images/fixed.nii")
fixedData = sitk.GetArrayFromImage(fixedData)

componentSegmentations = {}
numComponents = 8
imageDimensions = fixedData.shape
if len(imageDimensions) == 3:
    imageDepth = imageDimensions[0]
    imageWidth = imageDimensions[1]
    imageHeight = imageDimensions[2]
else:
    imageDepth = 1
    imageWidth = imageDimensions[0]
    imageHeight = imageDimensions[1]

for i in range(numComponents):
    temp = sitk.ReadImage("../images/segmentations/component"+str(i)+".nii")
    componentSegmentations[i] = sitk.GetArrayFromImage(temp)

print("Importing Image Data.")
movingImage = utils.normalizeImage(movingData)
fixedImage = utils.normalizeImage(fixedData)
for idx,img in componentSegmentations.items():
    componentSegmentations[idx] = utils.normalizeImage(img)

# utils.showNDA_InEditor_BW(movingData[10,:,:], "Moving Image")
# utils.showNDA_InEditor_BW(fixedData[10,:,:], "Fixed Image")
componentWeights = {}
for idx, img in componentSegmentations.items():
    # utils.showNDA_InEditor_BW(img[10,:,:], "Component " + str(idx))
    componentWeights[idx] = 1/numComponents # assume for now that these are fixed gamma terms

print("Normalizing Weights and Generating Weight Images.")
# STEP 2: Calculate the Normalized Weight Volume
weightImages = Weights.getNormalizedCommowickWeight(componentSegmentations,componentWeights)
# for idx, img in weightImages.items():
    # utils.showNDA_InEditor_BW(img[10,:,:], "Weight Image for Component "+ str(idx))

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

dim = _augmentDimensions(imageDimensions,[numComponents])

print("Composing Weight Volume")
weightVolume = np.zeros(dim,dtype=np.float64)
for idx in range(numComponents):
    weightImage = weightImages[idx]
    weightVolume[:,:,:,idx] = weightImage

# utils.showNDA_InEditor_BW(weightImages[1][10,:,:],"Weight Image")
# utils.showNDA_InEditor_BW(weightVolume[10,:,:,1], "Weight Volume Slice")

weightVolume = torch.tensor(data=weightVolume,dtype=torch.float64,requires_grad=False)
weightVolume = torch.reshape(weightVolume,shape=(imageWidth*imageHeight*imageDepth, numComponents))

print("Initializing Component Transforms.")
# STEP 3: Initialize transform variables
eye = torch.eye(4, dtype=torch.float64)
eye = eye.reshape((1,4,4))
componentTransforms = eye.repeat(numComponents,1,1)
componentTransforms = torch.autograd.Variable(data=componentTransforms,requires_grad=True)
# I stole the above pattern off the internet somewhere not sure entirely why it works
# NB: Pytorch3D uses row-ordinal matrices of format (batch, row, col)
# This means the affine matrices are reversed from their col-ordinal position!
# [ R 0 ]
# [ t 1 ]  If you put digits where the zero should be Pytorch3D throws an error.

print("Entering Registration Loop.")
# STEP 4: ENTER UPDATE LOOP
stop_loss = 1e-5
step_size = stop_loss / 3.0
maxItrs = 1

for i in range(maxItrs):
    print("\tCalculating Logs Mappings and Fusing...")
    logMaps = transforms.se3_log_map(componentTransforms)
    # The se3 log map takes the form [ R 1 | t 1 ] a [1 ,6] row matrix
    # This function returns N = numComponents row matrices.

    fusedVectorLogs = torch.matmul(weightVolume,logMaps)

    LEPTImageDimensions = _augmentDimensions(imageDimensions,[4,4])
    LEPTImageVolume = torch.zeros(LEPTImageDimensions,dtype=torch.float64)
    LEPTImageVolume = LEPTImageVolume.reshape((imageHeight*imageWidth*imageDepth,4,4))

    print("\tCalculating Exponential Mappings...")
    for i in range(imageWidth*imageHeight*imageDepth):
        LEPTImageVolume[i] = transforms.se3_exp_map(torch.reshape(fusedVectorLogs[i],(1,6)))

    LEPTImageVolume = LEPTImageVolume.reshape(LEPTImageDimensions)

    print("\tCalculating Displacements...")
    displacementFieldDimensions = _augmentDimensions(imageDimensions,len(imageDimensions))
    displacementField = torch.zeros(displacementFieldDimensions,dtype=torch.float64)

    if(len(imageDimensions) == 3):
        for depth in range(imageDimensions[0]):
            print("Processing slice ",depth + 1," of ",imageDepth,".")
            for row in range(imageDimensions[1]):
                for col in range(imageDimensions[2]):
                    homogeneousPoint = torch.tensor([depth,row,col,1],dtype=torch.float64)
                    newPoint = torch.matmul(LEPTImageVolume[depth,row,col],homogeneousPoint)
                    newPoint = torch.divide(newPoint,newPoint[len(imageDimensions)])[:len(imageDimensions)]
                    oldPoint = homogeneousPoint[:len(imageDimensions)]
                    displacementField[depth,row,col] = newPoint - oldPoint

# STEP 5: Warp Image
# Originally, we had discussed using SimpleITK.  There is an issue in that SimpleITK relies upon
# numpy, and not pytorch, to calculate the displacement and loss.  This means that the gradient
# propagation will be unhooked in the warping step and not able to propagate from the sitk loss
# calculation back into the model.

# TorchFields is a spacial warping package extending PyTorch for the express purpose of building
# spatial transformer networks.  Unfortunately, it has not been actively maintained for over 3
# years, and the feature set constrains itself to vector fields in two dimensions.  After examining
# their code, it appears that their warp field function (.sample()) is a warper around another
# pytorch function, grid_sample().  Grid sample is extensible to '5D' images (N,C,H,W,D) where
# N is the batch, C is the channels, and height, width, depth is the same.

with torch.no_grad():
    movingImage = sitk.GetImageFromArray(movingData)
    displacementField = displacementField.detach().numpy()
    disp = sitk.GetImageFromArray(displacementField.data)
'''
displacementField = sitk.GetImageFromArray(displacementField,isVector=True)
displacementField = sitk.DisplacementFieldTransform(displacementField)
movingImage = sitk.GetImageFromArray(movingImage.detach())
reference_image = movingImage
interpolator = sitk.sitkNearestNeighbor
default_value = 0.0
test = sitk.Resample(movingImage, reference_image, displacementField, interpolator, default_value)

utils.showNDA_InEditor_BW(test[32,:,:], "Output after 1 round.")
'''

def register():
    # Issue of ordering:
    # The order in which the weight masks are presented is the order in which the weights are assigned
    # The order in which the weights are assigned is the order in which the transforms are adjusted
    # The overall registration is agnostic to the position of the transforms (in terms of origin)
    # We only require the correct number of transforms and the weight masks determine the local effect

    # Create Normalized Weight Image
        # takes in N segmentation images
        # outputs 1 weight image volume with dimensions HxWxDxN

    # Initialize Transforms
    #       Set N 4x4 affine transforms to identity
    #       These are autograd variables

    # While loss > stop_loss & curr_itr < maxItrs:
    # _getLogTransforms()
    #   This is an N x 6 log mapping of the transform matrices
    # _getWeightedTransform()
    #   This takes the [N,6] log mapping matrix and HxWxDxN weight image
    #   weights it at every pixel respective to weight image
    #   Returns an HxWxDx6 matrix
    # _getLEPTVolume()
    #   Takes in the HxWxDx6 transform volume
    #   Returns the HxWxDx4x4 affine transformation volume
    # _getDisplacementField()
    #   Applies the affine transformation to each coordinate point in the image space as a homogeneous coordinate
    #   Returns the Cartesian vector (3x1) of displacement at each pixel
    # _getWarpedImage()
    #   Takes in the Warped Image and the Displacement Field and applies it to resample the image
    #   Output is the warped image
    # _update()
    #   Takes in the fixed image and the warped image
    #   Calculates and error metric
    #   Propagates the error back through transforms
    #   Updates Parameters
    # Returns N transformations estimated by the registrar algorithm
    pass

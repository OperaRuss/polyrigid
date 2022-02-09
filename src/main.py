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
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Custom Classes
import utilities as utils
from src import Weights

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

# STEP 1: Read in data
#   Our data comes in as .nii files pre-processed to reduce noise artifacts.
#   The subject is assumed to be the same for all images in the dataset (ie, MR images
#   of the same person).
movingData = sitk.ReadImage("../images/moving.nii")
movingData = sitk.GetArrayFromImage(movingData)
movingImage = torch.tensor(utils.normalizeImage(movingData),dtype=torch.float64)
movingImage = movingImage.unsqueeze(0).unsqueeze(0).permute((0, 1, 4, 3, 2))
# The above permutation needs to be done in order for the data to be in the format
# (Batch, Channels, Height, Width, Depth).  The moving image is only used in the
# calculation of the error metric under the warping parameters of the estimated
# displacement field.  This requires torch.grid_sample() which has a unique set of
# input requirements.

fixedData = sitk.ReadImage("../images/fixed.nii")
fixedData = sitk.GetArrayFromImage(fixedData)
fixedImage = torch.tensor(utils.normalizeImage(fixedData),dtype=torch.float64)

# Common Variable & Dimension definitions
numComponents = 8
imageDimensions = fixedData.shape
weightImageDimensions = _augmentDimensions(imageDimensions, [numComponents])
LEPTImageDimensions = _augmentDimensions(imageDimensions, [4, 4])
displacementFieldDimensions = _augmentDimensions(imageDimensions, len(imageDimensions))

if len(imageDimensions) == 3:
    imageDepth = imageDimensions[0]
    imageWidth = imageDimensions[1]
    imageHeight = imageDimensions[2]
else:
    imageDepth = 1
    imageWidth = imageDimensions[0]
    imageHeight = imageDimensions[1]

# Step ) Construct the weight image
componentSegmentations = {}

for i in range(numComponents):
    temp = sitk.ReadImage("../images/segmentations/component"+str(i)+".nii")
    componentSegmentations[i] = sitk.GetArrayFromImage(temp)

for idx,img in componentSegmentations.items():
    componentSegmentations[idx] = utils.normalizeImage(img)

componentWeights = {}
for idx, img in componentSegmentations.items():
    componentWeights[idx] = 1/numComponents # assume for now that these are fixed gamma terms

# STEP 2: Calculate the Normalized Weight Volume
weightImages = Weights.getNormalizedCommowickWeight(componentSegmentations, componentWeights)

# The weight volume is a matrix of dimensions [h,w,d,8] where each channel is the normalized
# weight image for one component. This is then reshaped into a [h*w*d,8] matrix for use in
# an inner product operation.
weightVolume = np.zeros(weightImageDimensions, dtype=np.float64)
for idx in range(numComponents):
    weightImage = weightImages[idx]
    weightVolume[:,:,:,idx] = weightImage

weightVolume = torch.tensor(data=weightVolume,dtype=torch.float64,requires_grad=False)
weightVolume = torch.reshape(weightVolume,shape=(imageWidth*imageHeight*imageDepth, numComponents))

# STEP 3: Initialize transform variables
# We are operating upon a static velocity field, where each point is assumed to operate
# locally as in Euclidean space.  We begin by initializing each component transform to
# the identity, which is a velocity of all zeros.  By taking the matrix exponential
# of a [4x4] zero matrix, we obtain the identity matrix for Euclidean space.
# This log-domain identity matrix is of the form [[ L v ], [000 1]] where
# L = rotation and v = translation.
eye = np.zeros((numComponents,4,4), dtype=np.float64) # zeros for [ L v ] matrix
for i in range(numComponents):
    eye[i] = np.eye(4)
eye = np.reshape(eye,(numComponents,16))
eye = torch.tensor(eye, requires_grad=True)
componentTransforms = torch.autograd.Variable(data=eye,requires_grad=True)
componentTransforms.retain_grad()

# STEP 4: ENTER UPDATE LOOP
stop_loss = 1e-5
step_size = 1e-3
maxItrs = 10
update_rate = 1
history = {}

# Create a regular sample grid in three dimensions
# Alternatively, we can normalize using a system of equations.  This is version pre-computes these values to save
# some flops where we can.
S_d = np.linspace(-1, 1, weightImageDimensions[0])
S_w = np.linspace(-1, 1, weightImageDimensions[1])
S_h = np.linspace(-1, 1, weightImageDimensions[2])

for itr in range(maxItrs):
    print("Beginning iteration ",itr,".")
    fusedVectorLogs = torch.matmul(weightVolume,componentTransforms)

    LEPTImageVolume = torch.zeros(LEPTImageDimensions,dtype=torch.float64)
    LEPTImageVolume = LEPTImageVolume.reshape((imageHeight*imageWidth*imageDepth,4,4))

    for i in range(imageWidth*imageHeight*imageDepth):
        LEPTImageVolume[i] = torch.matrix_exp(torch.reshape(fusedVectorLogs[i],(4,4)))

    LEPTImageVolume = LEPTImageVolume.reshape((20,64,64,4,4))

    displacementField = torch.zeros(displacementFieldDimensions,dtype=torch.float64)


    for depth in range(len(S_d)):
        for row in range(len(S_h)):
            for col in range(len(S_w)):
                homogeneousPoint = torch.tensor([S_d[depth],S_h[row],S_w[col],1],dtype=torch.float64)
                newPoint = torch.matmul(LEPTImageVolume[depth,row,col],homogeneousPoint)
                newPoint = torch.divide(newPoint,newPoint[len(imageDimensions)])[:len(imageDimensions)]
                oldPoint = homogeneousPoint[:len(imageDimensions)]
                displacement = oldPoint - newPoint
                displacementField[depth,row,col] = oldPoint + displacement

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

    displacementField = displacementField.unsqueeze(0)

    warped = F.grid_sample(movingImage,displacementField,mode='bilinear',padding_mode='zeros',align_corners=False)

    loss = F.mse_loss(warped.squeeze(),fixedImage)
    if itr % update_rate == 0:
        print("Loss at iteration ",itr,": ",loss)
        print("Component Transform",componentTransforms.data)

    loss.backward()

    # Recall that our affine matrices have been flattened into an [Nx16] matrix
    # Rotations in 3D would be [j,0:3,0:3] and translations [j,0:3,3]
    # Here we have to index them individually
    with torch.no_grad():
        fig = plt.imshow(warped[0,0,imageDepth//2,:,:])
        plt.axis('off')
        plt.title("Warped Image at iteration "+str(itr))
        plt.show()

        if itr % 2 == 0: # i is the iteration counter
            componentTransforms[:,0:3] -= torch.multiply(componentTransforms.grad[:,0:3],step_size)
            componentTransforms[:,4:7] -= torch.multiply(componentTransforms.grad[:,4:7],step_size)
            componentTransforms[:,8:11] -= torch.multiply(componentTransforms.grad[:,8:11],step_size)
        else:
            # Update translations for each component
            componentTransforms[:,3] -= torch.multiply(componentTransforms.grad[:,3],step_size)
            componentTransforms[:,7] -= torch.multiply(componentTransforms.grad[:,7],step_size)
            componentTransforms[:,11] -= torch.multiply(componentTransforms.grad[:,11],step_size)
        componentTransforms.grad.zero_()

    history[itr] = loss

    if loss < stop_loss:
        print("Model converged at iteration ",itr," with loss score ", loss)
        print("Normalized final parameters were ",componentTransforms)
        utils.showNDA_InEditor_BW(warped.detach().squeeze().numpy()[10,:,:],"Moving Image Result")
        utils.showNDA_InEditor_BW(fixedImage.numpy()[10,:,:],"Fixed Image Target")
        break

print("Congratulations, we did it!")
print("Final loss achieved: ",loss)

data = sorted(history.items())
x,y = zip(*data)
plt.plot(x,y)
plt.title("MSE Loss by Iterations")
plt.show()
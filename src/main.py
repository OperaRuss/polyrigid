'''
Russell Wustenberg, rw2873
polyrigid refactor
start date 20220220
'''

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utilities as utils

imgSITK_moving = sitk.ReadImage("../images/input/frame_5/frame_5.nii")
imgMoving = utils.normalizeImage(sitk.GetArrayFromImage(imgSITK_moving))
tImgMoving = torch.tensor(imgMoving, dtype=torch.float32)
tImgMoving = tImgMoving.unsqueeze(0).unsqueeze(0).cuda()

imgSITK_target = sitk.ReadImage("../images/input/frame_6/frame_6.nii")
imgTarget = utils.normalizeImage(sitk.GetArrayFromImage(imgSITK_target))
tImgTarget = torch.tensor(imgTarget, dtype=torch.float32)
tImgTarget = tImgTarget.unsqueeze(0).unsqueeze(0).cuda()

assert imgMoving.shape == imgTarget.shape, "Images must be of the same dimensions. Got %s != %s"%(imgMoving.shape,imgTarget.shape)

vNumComponents = 12
vImageDimensions = tImgTarget.shape
vNDims = len(vImageDimensions) - 2
vWeightImageDimensions = utils._augmentDimensions(vImageDimensions, [vNumComponents])
vLEPTImageDimensions_affine = utils._augmentDimensions(vImageDimensions, [vNDims + 1, vNDims + 1])
vLEPTImageDimensions_linear = (np.prod(vImageDimensions),vNDims + 1,vNDims + 1)
vDisplacementFieldDimensions = (*vImageDimensions[2:],vNDims)
vSamplePoints_Depth = torch.linspace(-1,1,steps=vImageDimensions[2])
vSamplePoints_Height = torch.linspace(-1,1,steps=vImageDimensions[3])
vSamplePoints_Width = torch.linspace(-1,1,steps=vImageDimensions[4])

tSamplePoints = torch.cartesian_prod(vSamplePoints_Depth,vSamplePoints_Height,vSamplePoints_Width)
tOnes = torch.ones(np.prod(vImageDimensions),dtype=torch.float32)
tSamplePoints = torch.cat((tSamplePoints,tOnes.unsqueeze(-1)),dim=1).cuda()

tTransform = torch.matmul(tSamplePoints,torch.eye(4).cuda())

tDisplacementField = torch.div(tTransform,tTransform[:,vNDims,None])[:,:vNDims]
tDisplacementField = torch.reshape(tDisplacementField,vDisplacementFieldDimensions).unsqueeze(0)

tImgWarped = F.grid_sample(tImgMoving,tDisplacementField,mode='bilinear',padding_mode='zeros',align_corners=False)
tImgFixed = F.grid_sample(tImgTarget,tDisplacementField,mode='bilinear',padding_mode='zeros',align_corners=False)

plt.imshow(tImgWarped.cpu().numpy()[0,0,vImageDimensions[2]//2,:,:])
plt.title("Warped Image")
plt.show()

plt.imshow(tImgFixed.cpu().numpy()[0,0,vImageDimensions[2]//2,:,:])
plt.title("Fixed Image")
plt.show()
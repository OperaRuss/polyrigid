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

vStop_Loss = .93
vStep_Size = .005
vMaxItrs = 100000
vUpdateRate = 100
vHistory = {}
vNumComponents = 12

imgSITK_moving = sitk.ReadImage("../images/input/cropped/frame_5/frame_5.nii")
imgMoving = utils.normalizeImage(sitk.GetArrayFromImage(imgSITK_moving))
tImgMoving = torch.tensor(imgMoving, dtype=torch.float32)
tImgMoving = tImgMoving.unsqueeze(0).unsqueeze(0).cuda()

imgSITK_target = sitk.ReadImage("../images/input/cropped/frame_6/frame_6.nii")
imgTarget = utils.normalizeImage(sitk.GetArrayFromImage(imgSITK_target))
tImgTarget = torch.tensor(imgTarget, dtype=torch.float32)
tImgTarget = tImgTarget.unsqueeze(0).unsqueeze(0).cuda()

assert imgMoving.shape == imgTarget.shape, \
    "Images must be of the same dimensions. Got %s != %s"%(imgMoving.shape,imgTarget.shape)

aComponentSegmentations = {}
aComponentWeightValues = {}

for i in range(vNumComponents):
    temp = sitk.ReadImage("../images/input/cropped/frame_5/frame_5_seg_comp_"+str(i)+".nii.gz")
    aComponentSegmentations[i] = sitk.GetArrayFromImage(temp)

for idx,img in aComponentSegmentations.items():
    aComponentSegmentations[idx] = utils.normalizeImage(img)

for idx, img in aComponentSegmentations.items():
    aComponentWeightValues[idx] = 1/2

vImageDimensions = tImgTarget.shape
vNDims = len(vImageDimensions) - 2
vComponentTransformDimensions = (vNumComponents,pow(vNDims + 1,2))
vWeightImageDimensions = (*vImageDimensions[2:],vNumComponents)
vLEPTImageDimensions_affine = utils._augmentDimensions(vImageDimensions, [vNDims + 1, vNDims + 1])
vLEPTImageDimensions_linear = (np.prod(vImageDimensions),vNDims + 1,vNDims + 1)
vDisplacementFieldDimensions = (*vImageDimensions[2::],vNDims)
tWeightVolume = np.zeros(vWeightImageDimensions)
tSamplePoints_Depth = torch.linspace(-1, 1, steps=vImageDimensions[2])
tSamplePoints_Height = torch.linspace(-1, 1, steps=vImageDimensions[3])
tSamplePoints_Width = torch.linspace(-1, 1, steps=vImageDimensions[4])

vWeightImages = utils._getWeightCommowick(aComponentSegmentations,aComponentWeightValues)

for idx in range(vNumComponents):
    if len(vImageDimensions) == 5:
        tWeightVolume[:,:,:,idx] = vWeightImages[idx]
    else:
        print("Not implemented for 2D at this time.")
        exit(2)

tWeightVolume = torch.tensor(tWeightVolume,dtype=torch.float32).cuda()

tSamplePoints = torch.cartesian_prod(tSamplePoints_Depth,tSamplePoints_Height, tSamplePoints_Width)
tOnes = torch.ones(np.prod(vImageDimensions),dtype=torch.float32)
tSamplePoints = torch.cat((tSamplePoints,tOnes.unsqueeze(-1)),dim=1).unsqueeze(-1).cuda()

tComponentTransforms = torch.zeros(vComponentTransformDimensions,dtype=torch.float32)
tComponentTransforms = torch.autograd.Variable(data=tComponentTransforms,requires_grad=True).cuda()
tComponentTransforms.retain_grad()
tViewTransform = utils.rotY(-np.pi/2.0) # Weird grid_sample bug requires a view transform to fix.
                                        # $50 to the first person to find where I went wrong.
for itr in range(vMaxItrs):
    tTransformField = torch.matmul(tWeightVolume,tComponentTransforms)
    tTransformField = torch.reshape(tTransformField,vLEPTImageDimensions_linear)
    tTransformField = torch.matrix_exp(tTransformField)
    tTransformField = torch.matmul(tViewTransform,tTransformField)
    tTransformField = torch.einsum('bij,bjk->bik',tTransformField,tSamplePoints)
    tTransformField.squeeze()

    tDisplacementField = torch.div(tTransformField,tTransformField[:,vNDims,None])[:,:vNDims]
    tDisplacementField = torch.reshape(tDisplacementField,vDisplacementFieldDimensions).unsqueeze(0)
    tDisplacementField = tDisplacementField

    tImgWarped = F.grid_sample(tImgMoving,tDisplacementField,
                               mode='bilinear',padding_mode='zeros',align_corners=False)
    tImgFixed = F.grid_sample(tImgTarget,tDisplacementField,
                              mode='bilinear',padding_mode='zeros',align_corners=False)

    loss = utils._getMetricNCC(tImgWarped,tImgFixed,5)
    loss.backward()

    with torch.no_grad():
        if vNDims == 3:
            if itr % 2 == 0:  # i is the iteration counter
                tComponentTransforms[:, 0:3] -= torch.multiply(tComponentTransforms.grad[:, 0:3], vStep_Size)
                tComponentTransforms[:, 4:7] -= torch.multiply(tComponentTransforms.grad[:, 4:7], vStep_Size)
                tComponentTransforms[:, 8:11] -= torch.multiply(tComponentTransforms.grad[:, 8:11], vStep_Size)
            else:
                # Update translations for each component
                tComponentTransforms[:, 3] -= torch.multiply(tComponentTransforms.grad[:, 3], vStep_Size)
                tComponentTransforms[:, 7] -= torch.multiply(tComponentTransforms.grad[:, 7], vStep_Size)
                tComponentTransforms[:, 11] -= torch.multiply(tComponentTransforms.grad[:, 11], vStep_Size)
        else:
            if itr % 2 == 0:  # rotation update
                tComponentTransforms[:, 0:2] -= torch.multiply(tComponentTransforms.grad[:, 0:2], vStep_Size)
                tComponentTransforms[:, 3:5] -= torch.multiply(tComponentTransforms.grad[:, 3:5], vStep_Size)
            else:  # translation update
                tComponentTransforms[:, 2] -= torch.multiply(tComponentTransforms.grad[:, 2], vStep_Size)
                tComponentTransforms[:, 5] -= torch.multiply(tComponentTransforms.grad[:, 5], vStep_Size)
        tComponentTransforms.grad.zero_()

    vHistory[itr] = abs(loss.item())

    if itr % vUpdateRate == 0:
        print("Loss at iteration ",itr,":",loss)

    if abs(loss) > vStop_Loss:
        print("Model converged at iteration ", itr, " with loss score ", loss)
        break

plt.imshow(tImgMoving.detach().squeeze().cpu().numpy()[vImageDimensions[2]//2, :, :],cmap='gray')
plt.title("Moving Image")
plt.show()

plt.imshow(tImgWarped.detach().squeeze().cpu().numpy()[vImageDimensions[2]//2, :, :],cmap='gray')
plt.title("Warped Image")
plt.show()

plt.imshow(tImgTarget.squeeze().cpu().numpy()[vImageDimensions[2]//2, :, :],cmap='gray')
plt.title("TargetImage")
plt.show()

sub = torch.subtract(tImgTarget, tImgWarped).squeeze().cpu()
plt.imshow(sub.detach().squeeze().numpy()[vImageDimensions[2]//2, :, :],cmap='gray')
plt.title("Target Image minus Warped Image")
plt.show()

data = sorted(vHistory.items())
x,y = zip(*data)
fig = plt.plot(x,y,marker =".",markersize=10)
plt.title("NCC Loss by Iterations")
plt.ylim(0.0,1.0)
plt.show()



with open("../images/results/20220222/res_transforms.txt","w") as out:
    print(f"MaxIterations: {vStep_Size}",file=out)
    print(f"Learning Rate: {vStep_Size}",file=out)
    print(f"Target Loss: {vStop_Loss:.4f}",file=out)
    print(f"Loss achieved: {loss:.4f}",file=out)
    print("Final parameter Estimations:\n",file=out)
    for i in range(vNumComponents):
        print("Component transformation "+str(i),file=out)
        print(torch.matrix_exp(torch.reshape(tComponentTransforms[i],(vNDims+1,vNDims+1))),file=out)

tImgWarped = sitk.GetImageFromArray(tImgWarped.detach().squeeze().cpu().numpy(),False)
sitk.WriteImage(tImgWarped, "../images/results/20220222/img_warped_latest.nii")

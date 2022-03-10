'''
Russell Wustenberg, rw2873
polyrigid refactor
start date 20220220

BEFORE RUNNING!
There are a number of I/O steps which are simply hard-coded in for rapid prototyping
at this time.  Make sure to change these things:
1) vDate  --  Change to today's date
2) Paths  --  Make sure the read in / write out paths are all correct.
              These are in the opening code block (after the variable definitions)
              and in the final code block (at the bottom of the program).  There
              are no intermitten i/o operations within the body of the code.
3) Results -- The program expects there to exist a 'results' folder in the 'images' folder.
              This is used for output of final charts.  Note that the error graph is NOT
              automatically saved at this time.

Note on interpreting error: the NCC function used here was developed by the Voxel Morph team
to massage NCC into a learning metric.  As such, they compute the NCC coefficients for all blocks
across the image (in a one-to-one mapping) and then take their average to produce an error score.
In order for the metric to be learnable, they add a negative sign BEFORE returning it.  So, the
error is bounded [0,1] but it will report [0,-1] with the error trending toward -1 with a better fit.
'''

import os
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utilities as utils
from sklearn.metrics import f1_score


# Model Parameters
# vMaxItrs and vLambda may be set to list objects in order to run mutiple consecutive runs using different parameters.
vStop_Loss = .93
vStep_Size = .005
vMaxItrs = [2] #[1,100,1000,10000]
vUpdateRate = 1000
vHistory = {}
vNumComponents = 12
vOutFile = "../images/results/"
vDate = "20220310"
vLambda = [0.0] #[1.0,0.75,0.5,0.25,0.0]

if not os.path.exists(vOutFile+vDate):
    os.makedirs(vOutFile+vDate)

imgSITK_moving = sitk.ReadImage("../images/input/cropped/frame_6/frame_6.nii")
imgMoving = utils.normalizeImage(sitk.GetArrayFromImage(imgSITK_moving))
tImgMoving = torch.tensor(imgMoving, dtype=torch.float32)
tImgMoving = tImgMoving.unsqueeze(0).unsqueeze(0).cuda()

imgSITK_target = sitk.ReadImage("../images/input/cropped/frame_5/frame_5.nii")
imgTarget = utils.normalizeImage(sitk.GetArrayFromImage(imgSITK_target))
tImgTarget = torch.tensor(imgTarget, dtype=torch.float32)
tImgTarget = tImgTarget.unsqueeze(0).unsqueeze(0).cuda()

assert imgMoving.shape == imgTarget.shape, \
    "Images must be of the same dimensions. Got %s != %s"%(imgMoving.shape,imgTarget.shape)

aComponentSegmentations_Float = {}
aComponentWeightValues = {}
aComponentSegmentations_Target = {}

for i in range(vNumComponents):
    temp = sitk.ReadImage("../images/input/cropped/frame_6/frame_6_seg_comp_"+str(i)+".nii.gz")
    aComponentSegmentations_Float[i] = sitk.GetArrayFromImage(temp)

    temp = sitk.ReadImage("../images/input/cropped/frame_5/frame_5_seg_comp_"+str(i)+".nii.gz")
    aComponentSegmentations_Target[i] = sitk.GetArrayFromImage(temp)

tComponentSegImg_Float = np.zeros(aComponentSegmentations_Float[0].shape, dtype=np.float32)
tComponentSegImg_Target = np.zeros(aComponentSegmentations_Target[0].shape, dtype=np.float32)

for idx,img in aComponentSegmentations_Float.items():
    tComponentSegImg_Float += aComponentSegmentations_Float[idx]
    tComponentSegImg_Target += aComponentSegmentations_Target[idx]
    aComponentSegmentations_Float[idx] = utils.normalizeImage(img)

'''
for i in range(9,12):
    plt.imshow(tComponentSegImg_Float[i,:,:],cmap='gray')
    plt.title("Float Slice "+str(i))
    plt.show()

    plt.imshow(tComponentSegImg_Target[i,:,:],cmap='gray')
    plt.title("Target Slice "+str(i))
    plt.show()
'''

for idx, img in aComponentSegmentations_Float.items():
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

vWeightImages = utils._getWeightCommowick(aComponentSegmentations_Float, aComponentWeightValues)

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


'''
The commented out line is an identiy velocity field (all zeros).  This caused learning
issues, and so I opted to initialize the components with small velocities drawn from a zero-centered
normal distribution.  It showed improvements over initializing at zero.

Also note that we have to manually set the final row to zeros so that it follows the appropriate
format to be a principal logarithm of an affine transformation matrix in homogeneous coordinates.
'''

#tComponentTransforms = torch.zeros(vComponentTransformDimensions,dtype=torch.float32)
tComponentTransforms = torch.normal(mean=0,std=torch.pi/(2e3),size=vComponentTransformDimensions)
tComponentTransforms[:,12:16] = 0
tComponentTransforms = torch.autograd.Variable(data=tComponentTransforms,requires_grad=True).cuda()
tComponentTransforms.retain_grad()
tViewTransform = utils.rotY(-np.pi/2.0,True) # Weird grid_sample bug requires a view transform to fix.
                                        # $50 to the first person to find where I went wrong.

#TEST VERISON
# If you desire to see how the model fuses known parameters, swap this code block out for the
# code block above.  Using the helper functions for rotations in the utilities file, you can
# compose Euclidean 4x4 transformation matricies and initialize the components that way.
# Setting vMaxItrs to 1 will force the LEPT fusion to only perform a forward pass through the model.
'''
import scipy

tComponentTransforms = np.zeros(vComponentTransformDimensions,dtype=np.float32)
for i in range(vNumComponents):
    temp = scipy.linalg.logm(utils.rotX((np.pi + np.random.normal(scale=0.5))/ 16.0, isTorch=False))
    tComponentTransforms[i,:] = np.reshape(temp,(1,16))
tComponentTransforms = torch.tensor(tComponentTransforms,dtype=torch.float32)
tComponentTransforms = torch.autograd.Variable(data=tComponentTransforms,requires_grad=True).cuda()
tComponentTransforms.retain_grad()
tViewTransform = utils.rotY(-np.pi/2.0,isTorch=True) # Weird grid_sample bug requires a view transform to fix.
                                        # $50 to the first person to find where I went wrong.
'''

for pMaxItrs in vMaxItrs:
    for pLamda in vLambda:
        print("Running with Smoothness Parameter " + str(pLamda)
              + " and JD Regularization parameter " + str(1.0 - pLamda))
        print("Running model for maximum of " + str(pMaxItrs) + " iterations.")
        for itr in range(pMaxItrs):
            tTransformField = torch.matmul(tWeightVolume,tComponentTransforms)
            tTransformField = torch.reshape(tTransformField,vLEPTImageDimensions_linear)
            tTransformField = torch.matrix_exp(tTransformField)
            tTransformField = torch.matmul(tViewTransform,tTransformField)
            # In the next step, we take the Einstein sum for dot prod along channel
            tTransformField = torch.einsum('bij,bjk->bik',tTransformField,tSamplePoints)
            tTransformField.squeeze()
            tTransformField = torch.div(tTransformField,tTransformField[:,vNDims,None])[:,:vNDims]

            tDisplacementField = torch.reshape(tTransformField,vDisplacementFieldDimensions).unsqueeze(0)
            tDisplacementField = tDisplacementField

            tImgWarped = F.grid_sample(tImgMoving,tDisplacementField,
                                       mode='bilinear',padding_mode='zeros',align_corners=False)
            tImgFixed = F.grid_sample(tImgTarget,tDisplacementField,
                                      mode='bilinear',padding_mode='zeros',align_corners=False)

            # loss = utils._getMetricMSE(tImgWarped,tImgFixed)
            loss = 1.0 * (utils._getMetricNCC(tImgWarped,tImgFixed,5)
                           + pLamda * utils._loss_Smooth(tDisplacementField)
                           + (1-pLamda) * utils._loss_JDet(tDisplacementField))
            loss.backward()

            with torch.no_grad():
                '''
                Here are two versions of the update function.  The one above (marked Experimental) attempts
                to leverage skew symmetry of rotation matrices on SO(3) to update the parameters.  By definition
                the rotation group consists of all unique rotations, which are defined by their skew symmetric form.
                The original Arsigny, et al., paper used skew symmetry to provide a unique fusion of velocities.  Later
                they moved to the principal matrix logarithm as a definition, since rotations will always be under pi radians
                in this definition.  Here, I was trying it out to see if it enforced orthogonality on the matrices.  It does
                show accuracy when close to the identity, but it does not produce meaningful results any more than the other
                rotation representation does.
                '''

                # Experimental Update function 3/4
                # skew symmetric matrix is S = A - A.T
                if itr % 2 == 0:
                    # Update Rotation
                    tUpdateTransforms = torch.matrix_exp(tComponentTransforms.view(vNumComponents,4,4)[:,0:3,0:3])
                    for i in range(vNumComponents):
                        tUpdateTransforms[i] = tUpdateTransforms[i].T * tComponentTransforms.grad.view(vNumComponents,4,4)[i,0:3,0:3]
                        tUpdateTransforms[i] = tUpdateTransforms[i] - tUpdateTransforms[i].T
                    tUpdateTransforms = torch.concat((tUpdateTransforms,torch.zeros((vNumComponents,3,1)).cuda()),axis=2)
                    tUpdateTransforms = torch.concat((tUpdateTransforms,torch.zeros((vNumComponents,1,4)).cuda()),axis=1)

                    tComponentTransforms -= torch.multiply(tUpdateTransforms.view(vNumComponents,16),vStep_Size)
                    tComponentTransforms.grad.zero_()
                else:
                    # Update translations for each component
                    tComponentTransforms[:, 3] -= torch.multiply(tComponentTransforms.grad[:, 3], vStep_Size)
                    tComponentTransforms[:, 7] -= torch.multiply(tComponentTransforms.grad[:, 7], vStep_Size)
                    tComponentTransforms[:, 11] -= torch.multiply(tComponentTransforms.grad[:, 11], vStep_Size)

                '''
                This is the original test function, following a standard format for gradient descent updates.
                Note that the affine matricies are stored in a [N,16] format so they must be handled wrt their
                positional arguments as though written as [[R t],[0000]] (and the last row must always be zeros
                in order for the principal matrix log to be well defined).
                
                UPDATE: 3/10 - initial tests in plotting the Jacobian determinant of the vector fields show
                that updating parameters in this way creates a higher rate of non-invertible transformations at
                each voxel.
                '''
                '''
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
                '''
            vHistory[itr] = loss.item()

            if itr % vUpdateRate == 0:
                print("Loss at iteration ",itr,":",loss)

            if abs(loss) > vStop_Loss:
                print("Model converged at iteration ", itr, " with loss score ", loss)
                break

            if itr > 10:
                if abs((loss + vHistory[itr-1] + vHistory[itr-2 ]) / 3.0) < 1e-5:
                    print("Model growth slowed at iteration", itr, "with loss score ", loss)
                    break

        tImgWarped_Seg = F.grid_sample(torch.tensor(tComponentSegImg_Float).cuda().unsqueeze(0).unsqueeze(0), tDisplacementField,
                                   mode='nearest', padding_mode='zeros', align_corners=False)

        str_smooth = str(pLamda).replace('.','_')
        str_JD = str(1.0-pLamda).replace('.','_')
        str_itrs = str(pMaxItrs)

        vSubFolder = "/obj_" + str(vNumComponents) + "_itrs_" + str_itrs + "_sm_" + str_smooth + "_JD_" + str_JD
        vPath = vOutFile + vDate + vSubFolder

        if not os.path.exists(vPath):
            os.makedirs(vPath)

        plt.imshow(tImgMoving.detach().squeeze().cpu().numpy()[vImageDimensions[2]//2, :, :],cmap='gray')
        plt.title("Moving Image")
        plt.axis('off')
        plt.savefig(vPath + "/img_moving.png",bbox_inches='tight')
        #plt.show()
        plt.close()

        plt.imshow(tImgWarped.detach().squeeze().cpu().numpy()[vImageDimensions[2]//2, :, :],cmap='gray')
        plt.title("Warped Image")
        plt.axis('off')
        plt.savefig(vPath + "/img_warped.png",bbox_inches='tight')
        #plt.show()
        plt.close()

        plt.imshow(tImgTarget.squeeze().cpu().numpy()[vImageDimensions[2]//2, :, :],cmap='gray')
        plt.title("Target Image")
        plt.axis('off')
        plt.savefig(vPath + "/img_target.png",bbox_inches='tight')
        #plt.show()
        plt.close()

        data = sorted(vHistory.items())
        x,y = zip(*data)
        fig = plt.plot(x,y,marker =".",markersize=10)
        plt.title("NCC Loss by Iterations")
        plt.ylim(0.0,1.0)
        plt.savefig(vPath + "/plot_NCC.png",bbox_inches='tight')
        #plt.show()
        plt.close()

        tImgWarped = sitk.GetImageFromArray(tImgWarped.detach().squeeze().cpu().numpy(),False)
        sitk.WriteImage(tImgWarped, vPath + "/nii_warped.nii")

        vDICE_Before = f1_score(tComponentSegImg_Target.reshape(-1,1),tComponentSegImg_Float.reshape(-1,1),average='macro')
        vDICE_After = f1_score(tComponentSegImg_Target.reshape(-1,1),tImgWarped_Seg.detach().cpu().view((-1,1)).numpy(),average="macro")

        tComponentSegImg_Float = sitk.GetImageFromArray(tComponentSegImg_Float,False)
        sitk.WriteImage(tComponentSegImg_Float,vPath + "/nii_float_seg.nii")

        tImgWarped_Seg = sitk.GetImageFromArray(tImgWarped_Seg.detach().squeeze().cpu().numpy(),False)
        sitk.WriteImage(tImgWarped_Seg,vPath + "/nii_warped_seg.nii")

        tComponentSegImg_Target = sitk.GetImageFromArray(tComponentSegImg_Target,False)
        sitk.WriteImage(tComponentSegImg_Target,vPath + "/nii_target_seg.nii")

        tDeterminantMap = utils.jacobian_determinant_3d(tDisplacementField)
        tDeterminantMap = tDeterminantMap.detach().cpu().numpy()
        tDeterminantMap = tDeterminantMap

        vLim = int
        if abs(tDeterminantMap.min()) >= abs(tDeterminantMap.max()):
            vLim = tDeterminantMap.min()
        else:
            vLim = tDeterminantMap.max()

        for i in range(tDeterminantMap.shape[0]):
            fig = plt.figure()
            plt.title("Determinant Map For Volume Slice "+str(i))
            im = plt.imshow(tDeterminantMap[i,:,:],cmap='bwr_r')
            plt.clim(-vLim,vLim)
            plt.axis('off')
            fig.colorbar(im)
            plt.savefig(vPath + "/img_det_slice_"+str(i)+".png", bbox_inches='tight')
            #plt.show()
            plt.close()

        vNumNeg = (tDeterminantMap <= 0.0).sum()

        tDeterminantMap = sitk.GetImageFromArray(tDeterminantMap,False)
        sitk.WriteImage(tDeterminantMap, vPath + "/nii_determinant.nii")

        with open(vPath + "/res_transforms.txt", "w") as out:
            print(f"MaxIterations: {vStep_Size}",file=out)
            print(f"Learning Rate: {vStep_Size}",file=out)
            print(f"Smoothness Parameter: {pLamda}",file=out)
            print(f"Jacobian Regularization Parameter: {(1.0-pLamda)}",file=out)
            print(f"DICE score before registration: {vDICE_Before:.4f}",file=out)
            print(f"DICE score after registration: {vDICE_After:.4f}",file=out)
            print(f"Target Loss: {vStop_Loss:.4f}",file=out)
            print(f"Loss achieved: {loss:.4f}",file=out)
            print(f"Percentage of Jacobian determinants negative: {(vNumNeg/(np.prod(vImageDimensions))*100):.2f}%",file=out)
            print("Final parameter Estimations:\n",file=out)
            for i in range(vNumComponents):
                print("Component transformation "+str(i),file=out)
                print(torch.matrix_exp(torch.reshape(tComponentTransforms[i],(vNDims+1,vNDims+1))),file=out)


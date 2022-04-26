import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

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

class Polyrigid(nn.Module):
    def __init__(self, imgFloat: torch.Tensor,
                 componentSegmentations: dict, componentWeights: dict):
        super().__init__()

        assert[len(imgFloat.shape) == 5,
               "Images must be in (N,C,D,H,W) format."]

        self.mNumComponents = len(componentSegmentations)
        self.mNDims = len(imgFloat.shape) - 2
        self.mImageDimensions = imgFloat.shape
        self.mComponentDims_Rotations = (self.mNumComponents, pow(self.mNDims,2))
        self.mComponentDims_Translations = (self.mNumComponents,self.mNDims)
        self.mComponentDims_Zeros = (self.mNumComponents,self.mNDims + 1)
        self.mWeightImageDimensions = (*self.mImageDimensions[2:],self.mNumComponents)
        self.mLEPTImageDimensions_affine = (*self.mImageDimensions,
                                            self.mNDims + 1, self.mNDims + 1)
        self.mLEPTImageDimensions_linear = (np.prod(self.mImageDimensions),
                                            self.mNDims + 1, self.mNDims + 1)
        self.mDisplacementFieldDimensions = (*self.mImageDimensions[2::],self.mNDims)

        self.tComponentRotations = nn.Parameter(torch.zeros(self.mComponentDims_Rotations).cuda(),
                                                requires_grad=True)
        self.tComponentTransforms = nn.Parameter(torch.zeros(self.mComponentDims_Translations).cuda(),
                                                 requires_grad=True)
        self.tDisplacementField = None
        self.tImgFloat = imgFloat
        self.tImgSegmentation = self._getSegmentationImage(componentSegmentations)
        self.tSamplePoints = self._getSamplePoints()
        self.tWeightVolume = self._getWeightVolume(componentSegmentations,
                                                   componentWeights)
        self.tViewTransform = rotY(-np.pi/2.0,True)

    def _getSegmentationImage(self,componentSegmentations):
        self.tImgSegmentation = np.zeros(self.mImageDimensions,dtype=np.float32)
        for label, img in componentSegmentations.items():
            if 1 <= label <= 8:
                self.tImgSegmentation += img
        return torch.tensor(self.tImgSegmentation,dtype=torch.float32).cuda()

    def _getSamplePoints(self):
        tSamplePoints_Depth = torch.linspace(-1, 1, steps=self.mImageDimensions[2])
        tSamplePoints_Height = torch.linspace(-1, 1, steps=self.mImageDimensions[3])
        tSamplePoints_Width = torch.linspace(-1, 1, steps=self.mImageDimensions[4])
        self.tSamplePoints = torch.cartesian_prod(tSamplePoints_Depth,
                                                  tSamplePoints_Height,
                                                  tSamplePoints_Width)
        tOnes = torch.ones(np.prod(self.mImageDimensions), dtype=torch.float32)
        return torch.cat((self.tSamplePoints, tOnes.unsqueeze(-1)), dim=1).unsqueeze(-1).cuda()

    def _getDistanceToCompoonentRegion(self, componentSegmentation: np.ndarray):
        '''
        :param componentSegmentation: Binary image with foreground values as objects and all else as background.
        :return: Returns the exact Euclidean distance from a background pixel to the nearest foreground pixel.
        '''
        maxIntensity = np.max(componentSegmentation)
        invertedImage = np.subtract(maxIntensity, componentSegmentation)
        return ndimage.distance_transform_edt(invertedImage)

    def _getRegionWeight(self, componentSegmentation: np.ndarray, gamma: float):
        '''
        :param componentSegmentation: Binary label image of a single component from the image.
        :param gamma: The relative weight assigned to this component.
        :return: Returns an imgae containing a diffusion of influence over the image space relative to the object.
        '''
        return (1.0 / (1.0 + (gamma * pow(self._getDistanceToCompoonentRegion(componentSegmentation), 2))))

    def _getWeightCommowick(self, componentSegmentations: dict, ratesOfDecay: dict):
        '''
        :param componentSegmentations: Dictionary of binary images for component regions
        :param ratesOfDecay: Dictionary of {label:weight} pairs where label = component segmentation label
        :return: Dictionary of normalized weight images summing to 1.0 at each voxel
        '''
        vCommowickWeights = {}
        for label, segmentation in componentSegmentations.items():
            vCommowickWeights[label] = self._getRegionWeight(segmentation, ratesOfDecay[label])
        vSumImage = np.zeros(componentSegmentations[next(iter(componentSegmentations))].shape, dtype=np.float32)
        for image in vCommowickWeights.values():
            vSumImage += image
        vNormalizedWeights = {}
        for label, image in vCommowickWeights.items():
            vNormalizedWeights[label] = np.divide(image, vSumImage)
        return vNormalizedWeights

    def _getWeightVolume(self, segmentations: dict, weights: dict):
        self.tWeightVolume = np.zeros(self.mWeightImageDimensions,dtype=np.float32)
        temp = self._getWeightCommowick(segmentations,weights)
        for i in range(1,self.mNumComponents):
            self.tWeightVolume[:,:,:,i] = temp[i]
        return torch.tensor(self.tWeightVolume,requires_grad=False).cuda()

    def _getLogComponentTransforms(self):
        temp = torch.cat((self.tComponentRotations,self.tComponentTransforms),axis=1)
        temp = torch.cat((temp,torch.zeros(self.mComponentDims_Zeros).cuda()),axis=1)
        return temp

    def _getLEPT(self,samplePoints=None):
        self.tDisplacementField = torch.matmul(self.tWeightVolume,
                                               self._getLogComponentTransforms())
        self.tDisplacementField = torch.reshape(self.tDisplacementField,
                                                self.mLEPTImageDimensions_linear)
        self.tDisplacementField = torch.matrix_exp(self.tDisplacementField)
        self.tDisplacementField = torch.matmul(self.tViewTransform,self.tDisplacementField)
        if samplePoints == None:
            self.tDisplacementField = torch.einsum('bij,bjk->bik',
                                                   self.tDisplacementField,self.tSamplePoints)
        else:
            self.tDisplacementField = torch.einsum('bij,bjk->bik',
                                                   self.tDisplacementField, samplePoints)
        self.tDisplacementField.squeeze()
        self.tDisplacementField = torch.div(self.tDisplacementField,
                                            self.tDisplacementField[:,self.mNDims,None])[:,:self.mNDims]
        return torch.reshape(self.tDisplacementField, self.mDisplacementFieldDimensions).unsqueeze(0)

    def _getLoss_Rigidity(self):
        transforms = self._getLogComponentTransforms()
        transforms = torch.matrix_exp(torch.reshape(transforms,
                                                   (self.mNumComponents,
                                                    self.mNDims + 1,
                                                    self.mNDims + 1)))
        transforms = transforms[:,0:3,0:3]
        sum = 0.0
        for i in range(self.mNumComponents):
            RRT = torch.sub(torch.matmul(transforms[i], transforms[i].T), torch.eye(3, device='cuda'))
            RTR = torch.sub(torch.matmul(transforms[i].T, transforms[i]), torch.eye(3, device='cuda'))
            Rdet = torch.det(transforms[i]) - 1.0
            sum += torch.frobenius_norm(RRT) + torch.frobenius_norm(RTR) + Rdet
        return sum

    def _getLoss_Translation_L1(self):
        transforms = self._getLogComponentTransforms()
        transforms = torch.matrix_exp(torch.reshape(transforms,
                                                    (self.mNumComponents,
                                                     self.mNDims + 1,
                                                     self.mNDims + 1)))
        transforms = transforms[:, 0:3, 3]
        return torch.mean(torch.sum(torch.abs(transforms),dim=1))

    def _getLoss_Translation_L2(self):
        transforms = self._getLogComponentTransforms()
        transforms = torch.matrix_exp(torch.reshape(transforms,
                                                    (self.mNumComponents,
                                                     self.mNDims + 1,
                                                     self.mNDims + 1)))
        transforms = transforms[:, 0:3, 3]
        return torch.mean(torch.sum(torch.multiply(transforms,transforms),dim=1))

    def _getLoss_DICE(self,tTargetSegmentation):
        tWarpedFloatSegmentation = F.grid_sample(self.tImgSegmentation,
                                   self.tDisplacementField,
                                   mode='bilinear', padding_mode='zeros',
                                   align_corners=False)
        sum_axes = list(range(2,self.mNDims + 2))
        numerator = 2 * (tWarpedFloatSegmentation * tTargetSegmentation).sum(dim=sum_axes)
        denominator = torch.clamp((tWarpedFloatSegmentation + tTargetSegmentation).sum(dim=sum_axes),min=1e-5)
        dice = torch.mean(numerator/denominator)
        return -dice

    def forward(self):
        self.tDisplacementField = self._getLEPT()
        return F.grid_sample(self.tImgFloat,self.tDisplacementField,
                             mode='bilinear',padding_mode='zeros',align_corners=False)
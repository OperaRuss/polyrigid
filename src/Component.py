import numpy as np
import SimpleITK as sitk
import utilities as utils
import Weights

class Component():
    def __init__(self,dimensions: int,rateOfDecay: float, segmentation: np.ndarray):
        '''
        A Component is the basic building block of the Polyrigid algorithm. Its primary functions are
        to possess the rotation and translation data for the component and to compose that data into a
        single affine matrix for use in the Polyrigid Registrar class.

        :param dimensions: Integer type indicating whether the image is a 2-dimensional or 3-dimensional image.
        :param rateOfDecay: Float value determining the rate of decay (or influence) of the component relative to the others in the image.
        :param segmentation: Numpy array immage of dtype=np.float64 containing the rigid component segmentation.
        '''
        self.mDimensions = dimensions
        self.mSegmentationImage = segmentation
        self.mNormalizedWeightImage = None
        self.mRotation = np.eye(dimensions,dtype=np.float64)
        self.mTranslation = np.zeros((dimensions,1),dtype=np.float64)
        self.mRateOfDecay = rateOfDecay

    def getAffineTransformMatrix(self):
        '''
        Function to produce an affine transformation matrix in homogenous coordinates for the given dimensionality.

        :return: 3x3 or 4x4 Numpy Array of format [[R t][0 1]], where R is a rotation matrix, t translation and last row of zeros with 1 in lower right corner.
        '''
        temp = np.concatenate((self.mRotation, self.mTranslation), axis=1)
        homoRow = np.concatenate((np.zeros(self.mDimensions),[1]), axis=0)
        affineMat = np.vstack((temp,homoRow))
        return affineMat

    def setUpdatedRotation(self, newRotation: np.ndarray):
        self.mRotation = newRotation

    def setUpdatedTranslation(self, newTranslation: np.ndarray):
        self.mTranslation = newTranslation

    def getWarpedSegmentation(self):
        '''
        Applies the selected component's current transform to its own segmentation image without any
        other input from the other components.  Included for possible testing purposes later in the process.

        :return: Returns an SITK Image containing the displaced segmentation.
        '''
        # todo Currently the displacement field is defined with affine matrices.  They need to be converted to vectors.

        transform = self.getAffineTransformMatrix()
        if (type(self.mSegmentationImage) != sitk.SimpleITK.Image):
            self.mSegmentationImage = sitk.GetImageFromArray(self.mSegmentationImage, False)
        dimensions = list(self.mSegmentationImage.shape)
        dimensions += [4,4]
        temp = np.zeros(tuple(dimensions),dtype=np.float64)
        temp[:,:] = transform
        displacementField = sitk.GetImageFromArray(temp,isVector=True)
        displacementField = sitk.DisplacementFieldTransform(displacementField)
        return utils.resampleImage(self.mSegmentationImage,displacementField)

class RigidComponents():
    def __init__(self, dimensions: int, ratesOfDecay: list,
                 componentSegmentations: list, imageDimensions: tuple=None):
        '''
        This is a container class for Component class objects.  Includes functionalities for normalizing
        weights given the set of provided components in the image. Automatically constructs components
        from the provided data (segmentations, etc.)

        :param dimensions: Integer value specifying whether the images are in 2 or 3 dimensions.
        :param ratesOfDecay: An in-order list of float type decay values for the components.
        :param componentSegmentations: In order list of numpy array images containing the binary mask of the rigid components.
        :param imageDimensions: Optional argument to pass the
        '''
        # todo add centroid-based definition of component with generated Gaussian weighting function

        self.mComponentList = []
        self.mImageDimensions = imageDimensions

        for idx in range(len(componentSegmentations)):
            self.mComponentList.append(Component(dimensions=dimensions, segmentation=componentSegmentations[idx],
                                                 rateOfDecay=ratesOfDecay[idx]))

    def setNormalizedComponentWeights(self):
        segmentations = []
        ratesOfDecay = []
        normalizedWeights = []

        for component in self.mComponentList:
            segmentations.append(component.mSegmentationImage)
            ratesOfDecay.append(component.mRateOfDecay)

        if self.mComponentList[0].mDimensions == 2:
            normalizedWeights = Weights.getNormalizedCommowickWeight(segmentations, ratesOfDecay, is3D=False)
        else:
            normalizedWeights = Weights.getNormalizedCommowickWeight(segmentations, ratesOfDecay, is3D=True)

        for idx in range(len(self.mComponentList)):
            self.mComponentList[idx].mNormalizedWeightImage = normalizedWeights[idx]


        # ToDo Add branching for isotropic vs anisotropic gaussian images
        # ToDo Figure out how to do in 3D...

        for idx in range(len(self.mComponentList)):
            self.mComponentList[idx].mNormalizedWeightImage = normalizedWeights[idx]

    def getWeightsAtCoordinate(self, Xcoord: int, Ycoord: int, Zcoord=None):
        '''
        Fetch function to return all component weights at a given pixel from their normalized weighting images.

        :param Xcoord: Row coordinate of the voxel
        :param Ycoord: Column coordinate of the voxel
        :param Zcoord: Depth coordinate of the voxel
        :return: A list of in-order component weights for the given pixel.
        '''
        componentWeights = []
        if len(self.mImageDimensions) == 2:
            for component in self.mComponentList:
                componentWeights.append(component.mNormalizedWeightImage[Xcoord][Ycoord])
        else:
            for component in self.mComponentList:
                componentWeights.append(component.mNormalizedWeightImage[Xcoord][Ycoord][Zcoord])
        return componentWeights

import copy
import SimpleITK as sitk
import numpy as np
import Component
from src import Weights


def loadNII(filePath: str):
    '''
    Function for loading an NIfTY1 image from file and converts it to SimpleITK format.

    :param filePath: File path to NIfTY1 image to load.
    :return: Returns a packed list: [SimpleITK image, Affine Transformation Matrix, NIfTY1 metadata header]
    '''
    return

class Image():
    '''
    Class assumes that it is provided a SimpleITK image in the .nii format (imported using the
    NiBabel python module.  It preserves the header information along with the original affine
    transformation information for re-assembly later.
    '''

    def __init__(self, imageData: np.ndarray, imageAffine: np.ndarray, imageHeader, dimensions: int=2):
        if dimensions != 2 and dimensions != 3:
            print("Images must be of dimensions 2 or 3 in the current implementation.")
            return

        self.mImage = imageData
        self.mAffine = imageAffine
        self.mHeader = imageHeader
        self.mDimensions = dimensions

    def getImageIntensities(self):
        return self.mImage

    def getImageAffine(self):
        return self.mAffine

    def getImageMetaData(self):
        return self.mHeader

    def getImageDimensions(self):
        return self.mDimensions


class FixedImage(Image):
    def __init__(self, imageData: np.ndarray, imageAffine: np.ndarray, imageHeader,
                 dimensions: int = 2):
        '''
        Currently, the only function of the fixed image is to pass all parameters on to the
        base image class.

        :param imageData: Intensity data for image to be warped.
        :param imageAffine: NII-type affine transformation matrix for adjustments.
        :param imageHeader: NIfTY1 metadata header preserved for repackaging after manipulation.
        :param imageComponents: A list of Component-class objects present in the image.
        :param dimensions: Integer indicating if te base image is 2D or 3D
        '''
        Image.__init__(self, imageData, imageAffine, imageHeader, dimensions)

class WarpedImage(Image):
    def __init__(self, imageData: np.ndarray, imageAffine: np.ndarray, imageHeader,
                 imageComponents: Component.RigidComponentBatchConstructor, dimensions: int=2):
        '''
        The Warped Image class provides a container for all data required to map the moving image onto the
        fixed image by use of a constructed displacement field.  The field is composed of vector displacements
        formed by a mixture of gaussians weighting of each rigid component.  Components should be provided as a
        list in the

        :param imageData: Intensity data for image to be warped.
        :param imageAffine: NII-type affine transformation matrix for adjustments.
        :param imageHeader: NIfTY1 metadata header preserved for repackaging after manipulation.
        :param imageComponents: A list of Component-class objects present in the image.
        :param dimensions:
        '''

        Image.__init__(self,imageData,imageAffine,imageHeader,dimensions)
        # todo Refactor container to make sure it works with the new dictionary structure
        self.mComponents = copy.deepcopy(imageComponents.mComponentDict)
        self.mWarpedImage = None
        self.setNormalizedComponentWeights()

    def setNormalizedComponentWeights(self):
        segmentations = {}
        ratesOfDecay = {}
        normalizedWeights = {}

        for label, component in self.mComponents.items():
            segmentations[component.getLabel()] = component.mSegmentationImage
            ratesOfDecay[component.getLabel()] = component.mRateOfDecay

        if self.mDimensions == 2:
            normalizedWeights = Weights.getNormalizedCommowickWeight(segmentations, ratesOfDecay)
        else:
            normalizedWeights = Weights.getNormalizedCommowickWeight(segmentations, ratesOfDecay)

        # ToDo Add branching for isotropic vs anisotropic gaussian images
        # ToDo Figure out how to do in 3D...

        for idx in range(len(self.mComponents)):
            self.mComponents[idx].mNormalizedWeightImage = normalizedWeights[idx]

    def getComponentTransforms(self):
        '''
        Returns all component transforms at a given pixel from their current affine transformation matrices.
        :return: Returns a square affine transformation matrix in a homogeneous coordinate system.
        '''
        componentTransforms = {}
        for label, component in self.mComponents.items():
            componentTransforms[label] = component.getAffineTransformMatrix()
        return componentTransforms

    def getWeightsAtCoordinate(self, Xcoord: int, Ycoord: int, Zcoord=None):
        '''
        Fetch function to return all component weights at a given pixel from their normalized weighting images.

        :param Xcoord: Row coordinate of the voxel
        :param Ycoord: Column coordinate of the voxel
        :param Zcoord: Depth coordinate of the voxel
        :return: A list of in-order component weights for the given pixel.
        '''
        componentWeights = {}
        for label, component in self.mComponents.items():
            componentWeights[label] = component.getWeightAtCoordinate(Xcoord,Ycoord,Zcoord)
        return componentWeights

    def getWarpedImage(self, displacementField: np.ndarray):

        def resampleImage(image: sitk.SimpleITK.Image, transform):
            '''
            This function was borrowed from the Kitware SimpleITK tutorial notebooks.

            :param image: A SimpleITK Image.
            :param transform: A displacement field transform with the same dimensions as the image.
            :return: The input image under the provided displacement transform.
            '''
            reference_image = image
            interpolator = sitk.sitkNearestNeighbor
            default_value = 0.0
            return sitk.Resample(image, reference_image, transform, interpolator, default_value)

        def warpImage(image, displacement):
            disp = sitk.GetImageFromArray(displacement, isVector=True)
            disp = sitk.DisplacementFieldTransform(disp)

            img = sitk.GetImageFromArray(image, False)
            post = resampleImage(img, disp)
            return post

        self.mWarpedImage = warpImage(self.mImage, displacementField)

        return self.mWarpedImage



def testFunction():
    import Shape

    print("Constructing Images.")
    moving, fixed, components = Shape.testImages()

    print("Constructing Moving Image.")
    mi = WarpedImage(moving,np.eye(4),None,components)

    print("Constructing Fixed Image.")
    fi = FixedImage(fixed,np.eye(4),None)

    return mi, fi
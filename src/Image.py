import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import Component

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


class WarpedImage(Image):
    def __init__(self,imageData: np.ndarray, imageAffine: np.ndarray, imageHeader,
                 imageComponents: Component.RigidComponents, dimensions: int=2):
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
        self.mComponents = imageComponents
        self.mComponents.setNormalizedComponentWeights()

    def getWarpedImage(self):
        # Figure out image shape
        # Tack on appropriate affine matrix (different for each case!)
        # create displacement field for vector image
        # populate by taking each pixel position and for each component multiplying it's affine transform by the weight

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

        def makeCartesian(point):
            temp = []
            if (type(point) == np.ndarray):
                tempPoint = point.tolist()
                temp = [x / tempPoint[-1] for x in tempPoint[:-1]]
            else:
                temp = [x / point[-1] for x in point[:-1]]
            return temp

        imageShape = self.mImage.shape
        transformMatrices = []

        for component in self.mComponents.mComponentList:
            transformMatrices.append(component.getAffineTransformMatrix())

        # Populate the displacement field
        if self.mDimensions == 2:
            imageShape = list(imageShape)
            imageShape += [3, 3]  # homogeneous coordinate affine matrix for a 2D image
            imageShape = tuple(imageShape)
            transformField = np.zeros(imageShape, dtype=np.float64)

            # Intensity Image Shape is (M,N,1)
            for row in range(self.mImage.shape[0]):
                for col in range(self.mImage.shape[1]):
                    coordWeights = self.mComponents.getWeightsAtCoordinate(row,col)
                    for idx in range(len(transformMatrices)):
                        transformField[row][col] += transformMatrices[idx] * coordWeights[idx]
        else:
            # Image is a 3D image
            pass

        # We now have the displacement Field, but it is composed of affine transformation matrices (not
        # displacement vectors.  We have to take the element-wise dot product of each of these matrices
        # with the original point vector.  This gives us the resulting sample point in the new image.
        # We then subtract the old from the new in order to find the forward displacement vectors.
        # We also must ensure that we de-homogenize the coordinates!

        # todo add a branching statement for 3D.
        imageShape = list(self.mImage.shape)
        imageShape += [2]  # 2D displacement vector for a 2D image
        imageShape = tuple(imageShape)

        displacementField = np.zeros(imageShape,dtype=np.float64)

        for row in range(self.mImage.shape[0]):
            for col in range(self.mImage.shape[1]):
                oldPoint = np.array([row,col,1])
                newPoint = np.dot(transformField[row][col],oldPoint)
                newPoint = makeCartesian(newPoint)

                displacementField[row][col] = np.subtract(newPoint, oldPoint[:-1])

        return warpImage(self.mImage,displacementField)

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

def testFunction():
    import Shape
    import utilities as utils

    print("Constructing Images.")
    moving, fixed, components = Shape.testImages()

    print("Constructing Moving Image.")
    mi = WarpedImage(moving,np.eye(4),None,components)

    print("Constructing Fixed Image.")
    fi = FixedImage(fixed,np.eye(4),None)

    return mi, fi
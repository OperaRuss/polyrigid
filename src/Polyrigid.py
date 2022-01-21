import scipy.linalg

import Image
import skimage
import SimpleITK as sitk
from scipy.linalg import logm, expm
import numpy as np
import utilities as utils



class PolyrigidRegistrar():
    def __init__(self, movingImage: Image.WarpedImage, targetImage: Image.FixedImage,
                 metric: str='JHMI', scalingFactor: int=10):
        '''
        This 'registrar' class encapsulates all necessary functions for
        :param movingImage:
        :param targetImage:
        :param metric:
        :param scalingFactor:
        '''
        self.mMovingImage = movingImage
        self.mTargetImage = targetImage
        self.mMetric = metric
        self.mScalingFactor = scalingFactor
        self.mMetricJHMIBins = 50
        self.mMetricRMSENormalization = 'euclidean'
        self.mLearningRates = [0.8,0.1,0.03]
        self.mMaxIterations = 1000
        self.mCurrIteration = 0

    def getMetric(self):
        if(self.mMetric == 'JHMI'):
            return skimage.metrics.normalized_mutual_information(sitk.GetArrayFromImage(self.mMovingImage.getWarpedImage()),
                                                                 self.mTargetImage.mImage,
                                                                 bins=self.mMetricJHMIBins)

        elif(self.mMetric == 'RMSE'):
            return skimage.metrics.normalized_root_mse(self.mMovingImage.getWarpedImage(),
                                       self.mTargetImage,
                                       self.mMetricRMSENormalization)
        else:
            print("Only Joint Hisotram Mutual Information (tag: 'JHMI') and Root Mean Square Error " +
                  "('RMSE') are implemented at this time.")

    def setMetric(self,metricName: str='JHMI',JHMIbins: int=50,RMSEnormalization: str='euclidean'):
        self.mMetric = metricName
        self.mMetricJHMIBins = JHMIbins
        self.mMetricRMSENormalization = RMSEnormalization

    def setMaxIterations(self, newMaxIters: int):
        self.mMaxIterations = newMaxIters

    def setLearningRates(self, newLearningRates: list):
        self.mLearningRates = newLearningRates

    def getMatrixLogarithm(self, affineTransformMatrix):
        '''
        Helper function to compute the matrix logarithm.  Uses Scipy Linalg module.
        Link to documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.logm.html

        :param affineTransformMatrix: A square affine transformation matrix in homogeneous coordinates
        :return: Returns the matrix logarithm computed by Pade approximants
        '''
        return logm(affineTransformMatrix)

    def getMatrixExponential(self, affineVelocityMatrix):
        '''
        Helper function to compute the matrix exponential.  Uses Scipy Linalg module.
        Link to documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html

        :param affineVelocityMatrix: A square affine velocity matrix in homogeneous coordinates (last row all zeros)
        :return: Returns the matrix exponential computed by Pade approximants
        '''
        return expm(affineVelocityMatrix)

    def getEmptyAffineTransformationImage(self, imageShape: tuple):
        '''
        :param imageShape: Tuple specifying a 2D or 3D image's dimensions
        :return: Returns an empty array with the same dimensions as the image and affine matrices at each pixel.
        '''
        if len(imageShape) == 2:
            imageShape = list(imageShape)
            imageShape += [3,3]
            imageShape = tuple(imageShape)
            transformImage = np.zeros(imageShape, dtype=np.float64)
        elif len(imageShape) == 3:
            imageShape = list(imageShape)
            imageShape += [4,4]
            imageShape = tuple(imageShape)
            transformImage = np.zeros(imageShape, dtype=np.float64)
        else:
            print("Error: Only 2D and 3D images are implemented for this program.")
            transformImage = None
        return transformImage

    def getEmptyDisplacementField(self,imageShape:tuple):
        '''
        :param imageShape: Tuple specifying a 2D or 3D image's dimensions.
        :return: Returns an empty vector field with 2D or 3D vectors at each pixel.
        '''
        if len(imageShape) == 2:
            imageShape = list(imageShape)
            imageShape += [2]
            imageShape = tuple(imageShape)
            displacementfield = np.zeros(imageShape, dtype=np.float64)
        elif len(imageShape) == 3:
            imageShape = list(imageShape)
            imageShape += [3]
            imageShape = tuple(imageShape)
            displacementfield = np.zeros(imageShape, dtype=np.float64)
        else:
            print("Error: Only 2D and 3D images are implemented for this program.")
            displacementfield = None
        return displacementfield

    def getLEPT(self, Xcoord: int, Ycoord: int, Zcoord: int=None):
        '''
        Function to return the composed log-Euclidean Polyaffine Transform matrix at a specific pixel location.
        :return: A square affine transformation matrix in homogeneous coordinates.
        '''
        weights = self.mMovingImage.getWeightsAtCoordinate(Xcoord,Ycoord,Zcoord)
        transforms = self.mMovingImage.getComponentTransforms()
        logTransforms = {}

        for component, transform in transforms:
            logTransforms[component] = self.getMatrixLogarithm(transform)

        compositeTransform = np.zeros(self.mMovingImage.mDimensions + 1, dtype=np.float64)

        for component, weight in weights:
            compositeTransform += weight * logTransforms[component]

        return self.getMatrixExponential(compositeTransform)

    def getDisplacementField(self):
        imageShape = self.mMovingImage.mImage.shape
        imageDimensions = self.mMovingImage.mDimensions
        transformImage = self.getEmptyAffineTransformationImage(imageShape)
        displacementField = self.getEmptyDisplacementField(imageShape)

        # populate the composite affine vectors at each pixel
        if imageDimensions == 2:
            for row in range(imageShape[0]):
                for col in range(imageShape[1]):
                    transformImage[row,col] = self.getLEPT(row,col)

            for row in range(imageShape[0]):
                for col in range(imageShape[1]):
                    point = utils.makeHomogeneous([row,col])
                    newPoint = np.dot(transformImage[row,col],point)
                    displacement = newPoint - point
                    displacementField[row,col] = utils.makeCartesian(displacement)
        else:
            for row in range(imageShape[0]):
                for col in range(imageShape[1]):
                    for depth in range(imageShape[2]):
                        transformImage[row,col,depth] = self.getLEPT(row,col,depth)

            for row in range(imageShape[0]):
                for col in range(imageShape[1]):
                    for depth in range(imageShape[2]):
                        point = utils.makeHomogeneous([row,col,depth])
                        newPoint = np.dot(transformImage[row,col,depth],point)
                        displacement = newPoint - point
                        displacementField[row,col] = utils.makeCartesian(displacement)

        return displacementField


    def register(self):
        '''
        This will eventually be the update loop which automatically calls the Registrar
        to map the moving image onto the fixed image.
        :return:
        '''
        # todo Figure out what data to return at the end for the next step of the pipeline
        # Next steps include: Iterating between images k+1 and k+2, Marching cubes on the segmentation images
        # Implies that we need to export (and save): warped segmentation images, transformation vectors
        # todo Would it be better to initialize successive rounds with the previously estimated vectors? ie, conservation of momentum?
        pass

def testFunction():
    import utilities as utils

    print("Starting.")
    moving, fixed = Image.testFunction()

    # Manually testing applying a rotation matrix to the warped images.
    a = PolyrigidRegistrar(moving,fixed)
    b = utils.getRotationMatrixFromRadians(utils.getRadiansFromDegrees(-20.0))
    c = utils.getRotationMatrixFromRadians(utils.getRadiansFromDegrees(10.0))
    a.mMovingImage.mComponents[0].setUpdatedRotation(b[:-1, :-1])
    a.mMovingImage.mComponents[1].setUpdatedRotation(c[:-1, :-1])

    import SimpleITK as sitk

    fixed = a.mTargetImage.mImage
    unwarped = a.mMovingImage.mImage
    warped = a.mMovingImage.getWarpedImage()
    compWarp = a.mMovingImage.mComponents[0].getWarpedSegmentation()
    utils.showNDA_InEditor_BW(fixed, "Fixed Test Image")
    utils.showNDA_InEditor_BW(unwarped, "Moving Test Image, Unwarped")
    utils.showNDA_InEditor_BW(sitk.GetArrayFromImage(warped), "Moving Test Image, Warped")
    utils.showNDA_InEditor_BW(sitk.GetArrayFromImage(compWarp), "Warped Segmentation")
    print(a.getMetric())

testFunction()

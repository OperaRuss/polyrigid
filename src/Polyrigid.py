import scipy.linalg

import Image
import skimage
import SimpleITK as sitk
from scipy.linalg import logm, expm

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

    def getLEPT(self):
        # todo Should the components store L & v vectors or R & t vectors
        #  Is the sequence:
        #       1) For each pixel,
#                       for each component, fetch the current transform matrix
        #       2)          For each component matrix, compute its logarithm
        #       3)      Sum these logarithmic matrices by weight
        #       4)      Exponentiate the summed matrix
#                   we now have the 'new T' for that pixel
#               5)  Warp image by new T vectors (by taking dot product with point vectors, etc)
        #   OR is this the sequence:
        #       1) For each pixel,
        #               for each component,
#                           fetch the stored VELOCITY matrix (all zeros on the bottom)
        #       2)      Scale these by a factor of 2^N to be near identity
        #       3)      Sum these together by weight
        #       4)      Eponentiate the summed matrix
        #       5)  We now have the 'new T' at each pixel, warp the image
        pass

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
    utils.showNDA_InEditor_BW(fixed, "Fixed Test Image")
    utils.showNDA_InEditor_BW(unwarped, "Moving Test Image, Unwarped")
    utils.showNDA_InEditor_BW(sitk.GetArrayFromImage(warped), "Moving Test Image, Warped")
    print(a.getMetric())
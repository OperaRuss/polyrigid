import Image
import skimage
import SimpleITK as sitk

class PolyrigidRegistrar():
    def __init__(self, movingImage: Image.WarpedImage, targetImage: Image.FixedImage,
                 metric: str='JHMI'):
        self.mMovingImage = movingImage
        self.mTargetImage = targetImage
        self.mMetric = metric
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
            print("Only Joing Hisotram Mutual Information (tag: 'JHMI') and Root Mean Square Error " +
                  "('RMSE') are implemented at this time.")

    def setMetric(self,metricName: str='JHMI',JHMIbins: int=50,RMSEnormalization: str='euclidean'):
        self.mMetric = metricName
        self.mMetricJHMIBins = JHMIbins
        self.mMetricRMSENormalization = RMSEnormalization

    def setMaxIterations(self, newMaxIters: int):
        self.mMaxIterations = newMaxIters

    def setLearningRates(self, newLearningRates: list):
        self.mLearningRates = newLearningRates

def testFunction():
    import utilities as utils

    print("Starting.")
    moving, fixed = Image.testFunction()

    # Manually testing applying a rotation matrix to the warped images.
    a = PolyrigidRegistrar(moving,fixed)
    b = utils.getRotationMatrixFromRadians(utils.getRadiansFromDegrees(-20.0))
    c = utils.getRotationMatrixFromRadians(utils.getRadiansFromDegrees(10.0))
    a.mMovingImage.mComponents.mComponentList[0].setUpdatedRotation(b[:-1, :-1])
    a.mMovingImage.mComponents.mComponentList[1].setUpdatedRotation(c[:-1, :-1])

    import SimpleITK as sitk
    unwarped = a.mMovingImage.mImage
    warped = a.mMovingImage.getWarpedImage()
    utils.showNDA_InEditor_BW(unwarped)
    utils.showNDA_InEditor_BW(sitk.GetArrayFromImage(warped))
    print(a.getMetric())

testFunction()

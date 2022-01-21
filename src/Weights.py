import numpy as np
import utilities as utils
from scipy import ndimage


def getDistanceToComponentRegion(componentSegmentation: np.ndarray):
    '''
    Comes from Makki, Borotikar, Garetier, et al., "A fast and memory-efficient algorithm for smooth
    interpolation of polyrigid transformations: application to human joint tracking." arXiv preprint
    arXiv:2005.02159. Source code found here: https://github.com/rousseau/dynMRI

    This function utilizes scipy.ndimage.distance_transform_edt() to calculate the distance
    between each pixel in the image and the region of interest.  The expectation is that
    the input has high values within the region of interest and zero (or negative) values elsewhere.

    :param componentSegmentation: A numpy array containing a segmentation of the region of interest
    :return: The distance from each pixel to the nearest in-region pixel.
    '''

    maxIntensity = np.max(componentSegmentation)
    invertedImage = np.subtract(maxIntensity, componentSegmentation)
    return ndimage.distance_transform_edt(invertedImage)

def getCommowickWeight(componentSegmentation: np.ndarray, gamma: float):
    '''
    This function comes from Commowick, Arsigny, Isambert, et al. "An efficient locally affine
    framework for the smooth registration of anatomical structures," Medical Image Analysis 12
    (2008), p427-441.

    :param componsnetSegmentation: Image as numpy array containing the binary label of
    :param gamma: User hyperparameter controlling rate of decay of component's region of influence.
    :return: The weight of contribution the component has at each pixel in the image.
    '''
    # Makki's implementation proposed an alternative function with steeper drop off
    # 2.0 / ( 1.0 + np.exp(0.4 * getDistanceToComponentRegion(componentSegmentation))
    return (1.0 / (1.0 + (gamma * pow(getDistanceToComponentRegion(componentSegmentation), 2))))

def getNormalizedCommowickWeight(componentSegmentations: dict, ratesOfDecay: dict):
    '''
    Function computes the normalized contribution weight at each pixel for each component. First,
    the raw contribution is contributed using the Commowick weight function.  Then, a sum is calculated
    over all contributions at each pixel.  Finally, each weight image is normalized by the sum of weights
    present at that pixel.  For pixels far from any component in the image, behavior may be unrealistic,
    and care must be taken when evaluating object behavior near the border of the region of interest.

    :param listComponentSegmentation: a list of images, as numpy arrays, containing component segmentation labels
    :param listParamsRateOfDecay: a list of float-type hyperparameters specifying the rate of decay of influence for each component
    :param is3D: boolean argument for whether images are 3D (True) or 2D (False).
    :return: a list of numpy arrays containing normalized contribution weight of each component
    '''

    commowickWeights = {}
    for label, segmentation in componentSegmentations:
        commowickWeights[label] = getCommowickWeight(segmentation, ratesOfDecay[label])
    
    imageDimensions = componentSegmentations[0].shape
    if(len(imageDimensions) == 3):
        sumImage = np.zeros((imageDimensions[0],imageDimensions[1],imageDimensions[2]),dtype=np.float64)

        for x in range(imageDimensions[0]):
            for y in range(imageDimensions[1]):
                for z in range(imageDimensions[2]):
                    for image in commowickWeights.values():
                        sumImage[x,y,z] += image[x,y,z]
    else:
        sumImage = np.zeros((imageDimensions[0],imageDimensions[1]), dtype=np.float64)

        for x in range(imageDimensions[0]):
            for y in range(imageDimensions[1]):
                for image in commowickWeights.values():
                    sumImage[x,y] += image[x,y]
    
    normalizedWeights = {}

    for label, image in commowickWeights:
        normalizedWeights[label] = np.divide(image, sumImage)
    
    return normalizedWeights

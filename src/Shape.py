import utilities as utils
import numpy as np

class Circle():
    def __init__(self, center: list, dimension: int=2):
        self.center = center
        self.radius = 1.0
        self.dimension = dimension
        self.matModel = np.identity(dimension + 1)
        self.matModelInv = np.identity(dimension + 1)

    def isWithin(self, point: list):
        if (self.dimension == 2):
            homoPoint = np.array(point)
            homoPoint = np.append(homoPoint,1)

            modelPoint = np.dot(self.matModelInv,homoPoint)
            modelPoint = np.divide(modelPoint, modelPoint[len(modelPoint) - 1])
            modelPoint = modelPoint[:-1]

            barycenter = np.array(self.center)
            barycenter = np.append(self.center, 1)
            barycenter = np.dot(self.matModel,barycenter)
            barycenter = np.divide(barycenter, barycenter[len(barycenter) - 1])
            barycenter = barycenter[:-1]

            radSqr = self.radius ** 2
            xSqr = (modelPoint[0] - barycenter[0]) ** 2
            ySqr = (modelPoint[1] - barycenter[1]) ** 2
            return (xSqr + ySqr) < radSqr
        elif (self.dimension == 3):
            radSqr = self.radius ** 2
            xSqr = (point[0] - self.center[0]) ** 2
            ySqr = (point[1] - self.center[1]) ** 2
            zSqr = (point[2] - self.center[2]) ** 2
            return (xSqr + ySqr + zSqr) < radSqr
        else:
            print("Dimension for the circle must be 2 or 3.")

    def updateInverseModelMatrix(self):
        self.matModelInv = np.linalg.inv(self.matModel)
    
    def scale(self, scaleFactor: float):
        self.radius *= scaleFactor
        '''
        if self.dimension == 2:
            self.matModel[0][0] *= scaleFactor
            self.matModel[1][1] *= scaleFactor
        elif self.dimension == 3:
            self.matModel[0][0] *= scaleFactor
            self.matModel[1][1] *= scaleFactor
            self.matModel[2][2] *= scaleFactor
        else:
            print("Scaling can only be done on circles of 2 and 3 dimensions currently.")
        self.updateInverseModelMatrix()
        '''

    def shear(self, shearFactorX: float=0, shearFactorY: float=0, shearFactorZ: float=0):
        if self.dimension == 2:
            scaleMat = np.eye(3)
            scaleMat[0][0] = shearFactorX
            scaleMat[1][1] = shearFactorY
            self.matModel = np.dot(scaleMat, self.matModel)
        elif self.dimension == 3:
            shearMat = np.eye(4)
            shearMat[1][0] = shearFactorY
            shearMat[2][0] = shearFactorZ
            shearMat[0][1] = shearFactorX
            shearMat[2][1] = shearFactorZ
            shearMat[0][2] = shearFactorX
            shearMat[1][2] = shearFactorY
            self.matModel = np.dot(shearMat,self.matModel)
        else:
            print("Scaling can only be done on circles of 2 and 3 dimensions currently.")
        self.updateInverseModelMatrix()
    
    def rotate(self, degrees):
        rotMat = utils.getRotationMatrixFromRadians(utils.getRadiansFromDegrees(degrees),self.dimension)
        self.matModel = np.dot(rotMat,self.matModel)
        self.updateInverseModelMatrix

def testFunction():
    img = np.zeros((64,64), dtype=np.float64)
    c = Circle([32,32])
    c.scale(6.0)
    c.shear(0.7,0.5)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(c.isWithin([x,y])):
                img[x,y] = 1.0

    utils.showNDA_InEditor_BW(img)

testFunction()
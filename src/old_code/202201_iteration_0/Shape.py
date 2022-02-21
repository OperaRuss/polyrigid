import matplotlib.pyplot as plt
import utilities as utils
import numpy as np


class Circle():
    def __init__(self, center: list, dimension: int=2):
        '''
        Horribly flawed class to draw an ellipsoid in 2 and 3 dimensions.  Attempted to do so using
        an implicit equation, and it ended up unable to rotate.  Useful for drawing basic circles and ellipses
        that undergo translation only.

        :param center: A list type of integers [Xcoord, Ycoord, etc]
        :param dimension: Number of dimensions of the shape being drawn
        '''

        self.center = center
        self.radius = 1.0
        self.dimension = dimension
        self.matModel = np.identity(dimension + 1)
        self.matModelInv = np.identity(dimension + 1)

    def isWithin(self, point: list):
        '''
        A basic ray tracing function.  Uses the array indices as point samples into the continuous object space.
        If a pixel satisfies the equation, it is set to full on (1.0), else it is full off.

        :param point: Point to be queried
        :return: Bool type return of whether the coordinate satisfies the circle's implicit equation.
        '''
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

    def shear(self, shearFactorX: float=0, shearFactorY: float=0, shearFactorZ: float=0):
        '''
        Useful for drawing ellipses, but they don't turn out very well under rotation.

        :param shearFactorX:
        :param shearFactorY:
        :param shearFactorZ:
        '''
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
        '''
        Applies a rotation matrix to the object, but does so about the image origin at the current time.
        Could be useful if the whole class were to be refactored.
        :param degrees:
        :return:
        '''
        rotMat = utils.getRotationMatrixFromRadians(utils.getRadiansFromDegrees(degrees),self.dimension)
        self.matModel = np.dot(rotMat,self.matModel)
        self.updateInverseModelMatrix()

def testFunction():
    img1 = np.zeros((64,64), dtype=np.float64)
    c = Circle([32,32])
    c.scale(15.0)

    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            if(c.isWithin([x,y])):
                img1[x,y] = 1.0

    utils.normalizeForImageOutput(img1)
    utils.saveNPimage_2D_BW(img1,"test_moving.png",r"C:\Users\russe\github\polyrigid\images\test")

    img2 = np.zeros((64, 64), dtype=np.float64)
    c = Circle([22, 42])
    c.scale(15.0)

    for x in range(img2.shape[0]):
        for y in range(img2.shape[1]):
            if (c.isWithin([x, y])):
                img2[x, y] = 1.0

    utils.normalizeForImageOutput(img2)
    utils.saveNPimage_2D_BW(img2, "test_fixed.png", r"C:\Users\russe\github\polyrigid\images\test")

def testImages():
    movingImg = np.zeros((64, 64), dtype=np.float64)
    seg1 = np.zeros((64,64), dtype=np.float64)
    seg2 = np.zeros((64,64), dtype=np.float64)
    c1 = Circle([42, 42])
    c1.scale(15.0)
    c2 = Circle([10,10])
    c2.scale(7.0)

    for x in range(movingImg.shape[0]):
        for y in range(movingImg.shape[1]):
            if c1.isWithin([x,y]):
                movingImg[x,y] = 1.0
                seg1[x,y] = 1.0
            if c2.isWithin([x,y]):
                movingImg[x,y] = 1.0
                seg2[x,y] = 1.0

    import Component

    objs = [c1, c2]
    segs = [seg1, seg2]
    infl = [0.5,0.5]

    components = Component.RigidComponentBatchConstructor(2, infl, segs)

    fixedImage = np.zeros((64, 64), dtype=np.float64)
    c1.center = [40,35]
    c2.center = [15,20]
    for x in range(movingImg.shape[0]):
        for y in range(movingImg.shape[1]):
            if c1.isWithin([x,y]):
                fixedImage[x,y] = 1.0
            if c2.isWithin([x,y]):
                fixedImage[x,y] = 1.0

    return movingImg, fixedImage, components
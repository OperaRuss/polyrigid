import utilities as utils
import numpy as np

class Circle():
    def __init__(self, center: list, radius: float, dimension: int=2):
        self.center = center
        self.radius = radius
        self.dimension = dimension
        self.matModel = np.identity(dimension + 1)

    def isWithin(self, point: list):
        if (self.dimension == 2):
            worldCenter = np.dot(self.matModel,self.center)
            radSqr = self.radius ** 2
            xSqr = (point[0] - worldCenter[0]) ** 2
            ySqr = (point[1] - worldCenter[1]) ** 2
            return (xSqr + ySqr) < radSqr
        elif (self.dimension == 3):
            radSqr = self.radius ** 2
            xSqr = (point[0] - self.center[0]) ** 2
            ySqr = (point[1] - self.center[1]) ** 2
            zSqr = (point[2] - self.center[2]) ** 2
            return (xSqr + ySqr + zSqr) < radSqr
        else:
            print("Dimension for the circle must be 2 or 3.")

class Ellipse():
    def __init__(self, center: list, axisMajor: float, axisMinor: float, axisWonky: float=1.0, dimension: int=2):
        self.center = center
        self.axisMajor = axisMajor
        self.axisMinor = axisMinor
        self.axisWonky = axisWonky
        self.dimension = dimension

    def isWithin(self, point:list):
        epsilon = 1e-5

        if (self.dimension == 2):
            majRadSqr = self.axisMajor ** 2
            minRadSqr = self.axisMinor ** 2
            xSqr = (point[0] - self.center[0]) ** 2
            ySqr = (point[1] - self.center[1]) ** 2
            return ((xSqr / (majRadSqr + epsilon)) +
                    (ySqr / (minRadSqr + epsilon))) <= 1
        elif (self.dimension == 3):
            majRadSqr = self.axisMajor ** 2
            minRadSqr = self.axisMinor ** 2
            wokRadSqr = self.axisWonky ** 2
            xSqr = (point[0] - self.center[0]) ** 2
            ySqr = (point[1] - self.center[1]) ** 2
            zSqr = (point[2] - self.center[2]) ** 2
            return ((xSqr/(majRadSqr + epsilon)) + \
                   (ySqr/(minRadSqr + epsilon)) + \
                   (zSqr/(wokRadSqr + epsilon))) <= 1
        else:
            print("Dimensionality for the ellipse must be either 2 or 3.")
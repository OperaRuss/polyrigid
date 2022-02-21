import copy

import pytorch3d.transforms

import numpy as np
import utilities as utils


def numpyFunk():
    t1_R = utils.getRotationMatrixFromRadians((np.pi/16.0),2)
    t1_R = np.array(t1_R, dtype=np.float64)

    t2_R = utils.getRotationMatrixFromRadians((np.pi/-16.0),2)
    t2_R = np.array(t2_R, dtype=np.float64)

    dim = (400,400)

    comp1 = np.zeros(dim,np.float64)
    coordY = int(0.25 * dim[0])
    coordX = int(0.5 * dim[1])
    m = [coordX,coordY]
    s = [[0.5,0],[0,0.5]]

    centroid1 = copy.deepcopy(m)

    def multivariateGaussian(x, mu, sigma):
        k = len(mu)
        invCov = np.linalg.pinv(sigma)
        div = (2 * (np.pi)) ** k
        div = div * np.linalg.det(sigma)
        div = 1 / (np.sqrt(div))
        exp = -0.5 * (np.dot(np.subtract(x,mu),np.dot(invCov,np.subtract(x,mu))))
        return div * exp

    for row in range(comp1.shape[0]):
        for col in range(comp1.shape[1]):
            comp1[row,col] = multivariateGaussian([row, col], m, s)


    comp2 = np.zeros(dim,np.float64)
    coordY = int(0.75 * dim[1])
    m = [coordX,coordY]
    s = [[0.5,0],[0,0.5]]
    centroid2 = copy.deepcopy(m)

    for row in range(comp2.shape[0]):
        for col in range(comp2.shape[1]):
            comp2[row,col] = multivariateGaussian([row,col],m,s)

    comp1 = utils.normalizeImage(comp1)
    comp2 = utils.normalizeImage(comp2)
    sum = np.add(comp1,comp2)

    utils.showNDA_InEditor_BW(comp1, "Comp 1")
    utils.showNDA_InEditor_BW(comp2, "Comp 2")
    utils.showNDA_InEditor_BW(sum, "Sum")

    normWeight1 = np.divide(comp1,sum)
    normWeight2 = np.divide(comp2,sum)

    weights = [normWeight1, normWeight2]

    weightShape = normWeight1.shape
    weightShape = list(weightShape)
    weightShape.append(len(weights))
    weightShape = tuple(weightShape)


    weightImageVolume = np.dstack((normWeight1,normWeight2))

    utils.showNDA_InEditor_BW(weightImageVolume[:,:,0], "weight 1")
    utils.showNDA_InEditor_BW(weightImageVolume[:,:,1], "weight 2")

    temp1 = np.reshape(t1_R,(9,1))
    temp2 = np.reshape(t2_R,(9,1))
    t3 = np.concatenate((temp1,temp2),axis=1)

    def getDimPlus(dim, n):
        temp = list(dim)
        temp.append(n)
        return tuple(temp)

    newT = np.zeros(getDimPlus(dim,9))

    for row in range(newT.shape[0]):
        for col in range(newT.shape[1]):
            newT[row,col] = np.dot(t3,weightImageVolume[row,col])

    displacements = np.zeros(getDimPlus(dim,3))

    for row in range(displacements.shape[0]):
        for col in range(displacements.shape[1]):
            displacements[row,col] = np.dot(np.reshape(newT[row,col],(3,3)),np.array([row,col,1]))

    dispCartesian = np.zeros(getDimPlus(dim,2))

    for row in range(displacements.shape[0]):
        for col in range(displacements.shape[1]):
            dispCartesian[row,col] = np.divide(displacements[row,col,:2],displacements[row,col,2])


    displacementField = np.zeros(dispCartesian.shape)

    for row in range(displacements.shape[0]):
        for col in range(displacements.shape[1]):
            displacementField[row,col] = np.subtract(dispCartesian[row,col],np.array([row,col]))

    import SimpleITK as sitk
    disp = sitk.GetImageFromArray(displacementField, isVector=True)
    disp = sitk.DisplacementFieldTransform(disp)

    img = np.zeros(dim)
    thresh = int(dim[0] / 20.0)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if row < thresh or row > (img.shape[0] - thresh) \
                    or col < thresh or col > (img.shape[1] - thresh):
                pass
            elif row % 4 == 0 or col % 4 == 0:
                img[row,col] = 1

    utils.showNDA_InEditor_BW(img)

    img = sitk.GetImageFromArray(img, isVector=False)
    reference_image = img
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0.0
    result = sitk.Resample(img, reference_image, disp, interpolator, default_value)
    utils.showSITK_InEditor_BW(result, "result without log Euclidean")

def torchTestFunc():
    import torch

    A = torch.tensor(torch.rand((3,3)),requires_grad=False)
    w1 = torch.autograd.Variable(torch.rand((3,3)), requires_grad=True)
    w2 = torch.autograd.Variable(torch.rand((3,3)), requires_grad = True)

    weights = [w1,w2]

    stop_loss = 1e-5
    step_size = stop_loss/3.0

    history = {}

    import scipy.linalg

    def adjoint(A, E, f):
        A_H = A.T.conj().to(E.dtype)
        n = A.size(0)
        M = torch.zeros(2*n, 2*n, dtype=E.dtype, device=E.device)
        M[:n, :n] = A_H
        M[n:, n:] = A_H
        M[:n, n:] = E
        return f(M)[:n, n:].to(A.dtype)

    def logm_scipy(A):
        return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)

    class Logm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A):
            assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
            assert A.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
            ctx.save_for_backward(A)
            return logm_scipy(A)

        @staticmethod
        def backward(ctx, G):
            A, = ctx.saved_tensors
            return adjoint(A, G, logm_scipy)

    for i in range (100000):
        w = torch.zeros(weights[0].shape) # dummy variable
        for weight in weights:
            logw = logm_scipy(weight)
            print(logw.data)
            w += torch.multiply(logw,1/len(w))
        w = torch.matrix_exp(w)
        f = torch.matmul(A,w)
        l = torch.norm(f,p=2)
        l.backward()
        if i % 2 == 0:
            w1.data -= step_size * w1.grad.data
        else:
            w2.data -= step_size * w2.grad.data
        w1.grad.zero_()
        w2.grad.zero_()
        if i % 1000 == 0:
            print("Loss at iteration ",i,": ",l)
            history[i] = l.item()
        if abs(l) < stop_loss:
            print("Final loss at iteration ",i,": ",l)
            break

    import matplotlib.pyplot as plt
    lists = sorted(history.items())
    x,y = zip(*lists)
    plt.plot(x,y)
    plt.show()

def torchCompositionTest():
    import torch
    import scipy.linalg
    import matplotlib.pyplot as plt

    def adjoint(A, E, f):
        A_H = A.T.conj().to(E.dtype)
        n = A.size(0)
        M = torch.zeros(2*n, 2*n, dtype=E.dtype, device=E.device)
        M[:n, :n] = A_H
        M[n:, n:] = A_H
        M[:n, n:] = E
        return f(M)[:n, n:].to(A.dtype)

    def logm_scipy(A):
        return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)

    class Logm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A):
            assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
            assert A.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
            ctx.save_for_backward(A)
            return logm_scipy(A)

        @staticmethod
        def backward(ctx, G):
            A, = ctx.saved_tensors
            return adjoint(A, G, logm_scipy)


    # Declare Variables
    r1 = torch.autograd.Variable(torch.eye(3), requires_grad=True)
    r2 = torch.autograd.Variable(torch.eye(3), requires_grad=True)
    point = torch.tensor(torch.rand(3,1), requires_grad=False)
    theta = np.pi / 16.0
    true_rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],[np.sin(theta),np.cos(theta),0.],[0.,0.,1.]],dtype=np.float64)
    target = torch.tensor(np.dot(true_rotation,point.data), requires_grad=False)

    rotations = [r1,r2]

    def composeLERotationalT(rotations: list):
        temp = torch.zeros((3,3))
        for rot in rotations:
            temp += logm_scipy(rot)
        return torch.matrix_exp(temp)

    stop_err = 1e-2
    step_size = stop_err / 3.0

    import torch.nn as nn
    mse_loss = nn.MSELoss()

    history = {}

    def plotHistory(history):
        lists = sorted(history.items())
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.show()

    maxIters = 100

    for i in range(maxIters):
        newT = composeLERotationalT(rotations)
        func = torch.matmul(newT,point)
        loss = mse_loss(func,target)
        loss.backwards()
        for rot in rotations:
            rot -= rot.grad.data * step_size
            rot.grad.zero_()
        if loss < stop_err:
            print("Loss converged at iteration ",i," with MSE loss score of ",loss)
            history[i] = loss
        elif i % int(maxIters / 20.0):
            print("Loss at iteration ",i," with loss score\t",loss)
            history[i] = loss

    plotHistory(history)

def tfTestFunc():
    import os
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin")
    os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v8.3\bin")

    import tensorflow as tf
    A = tf.Variable(tf.eye(3,3), name='A')
    B = tf.Variable(tf.eye(3,3), name='B')
    x = tf.constant(tf.random.uniform((3,1)))
    y = tf.constant(tf.random.uniform((3,1)))

    stop = 1e-2
    step = stop / 3.0
    maxItrs = 10000
    history = {}

    for i in range(maxItrs):
        newT = tf.multiply(A,0.5) + tf.multiply(B,0.5)
        with tf.GradientTape() as tape:
            y_hat = tf.matmul(newT,x)
            loss = tf.math.reduce_mean(tf.math.squared_difference(y_hat,y))

        dL_dT = tape.gradient(loss,newT)
        dL_dA = tape.gradient(newT,A)
        dL_dB = tape.gradient(newT,B)
        A.assign_sub(dL_dA * step)
        B.assign_sub(dL_dB * step)
        if abs(loss) < stop:
            print("Loss converged at iteration ", i, " with MSE loss score of ", loss)
            history[i] = loss
            break
        elif i % int(maxItrs / 20.0):
            print("Loss at iteration ", i, " with loss score\t", loss)
            history[i] = loss

    import matplotlib.pyplot as plt
    def plotHistory(history):
        lists = sorted(history.items())
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.show()

    plotHistory(history)
    print(A)

def torch3Dtest():
    import torch
    from pytorch3d import transforms
    import torch.nn


    A = torch.eye(4)
    B = torch.eye(4)
    B = torch.multiply(B,np.cos(np.pi/8.0))
    B[3,3] = 1.
    B[3,:3] = 25.
    A = A.reshape((1,4,4))
    A = A.repeat(4,1,1)
    A[1] = B
    A[3,3,0:3] = -3.2
    A[:,:3,0:3] = pytorch3d.transforms.random_rotation()
    print(A)
    A = torch.autograd.Variable(A, requires_grad=True)
    logs = transforms.se3_log_map(A)
    print(logs)
    weights = torch.tensor(torch.rand((2,4,1)),requires_grad=False)

    def _composeNewT(mat,weights):
        temp = torch.matmul(mat.T,weights)
        return temp.T
    print(_composeNewT(logs,weights))

    summ = _composeNewT(logs,weights)

    exp = transforms.se3_exp_map(summ)
    exp[:3,3] -= 3.
    loss = torch.nn.MSELoss()
    l = loss(A,exp)
    l.backward()
    print(l)
    print(A.grad)

def reshape_practice():
    import torch

    N = 8
    h = 10
    w = 10
    d = 10

    a = torch.rand((N,6))
    b = torch.rand((h,w,d,N))
    b_prime = torch.reshape(b,(h*w*d,N))
    c = torch.matmul(a.T,b_prime.T)
    print(c.shape)
    print(c.T.shape)
    c = torch.reshape(c,(h,w,d,6))
    print(c)

def grid_sample_practice():
    import torch
    import torch.nn.functional as F

    dim = [100, 100]

    normCoords = [[2.0 / dim[0], 0, -((dim[0] - 1) / dim[0])],
                  [0, 2.0 / dim[1], -((dim[0] - 1) / dim[0])],
                  [0, 0, 1]]

    # normCoords = np.linalg.inv(normCoords)

    normCoords = torch.tensor(normCoords, dtype=torch.float32)

    dim = [1, 1] + dim
    dim = tuple(dim)

    moving = torch.zeros(dim, dtype=torch.float32)
    fixed = torch.zeros(dim, dtype=torch.float32)

    print(moving.shape)
    for i in range(dim[-2]):
        for j in range(dim[-1]):
            if i > dim[-2] * .25 and i < dim[-2] * .75:
                if j > dim[-1] * .25 and j < dim[-1] * .75:
                    moving[0, 0, i, j] = 1.0

    for i in range(dim[-2]):
        for j in range(dim[-1]):
            if i > dim[-2] * .35 and i < dim[-2] * .85:
                if j > dim[-1] * .35 and j < dim[-1] * .85:
                    fixed[0, 0, i, j] = 1.0

    utils.showNDA_InEditor_BW(moving.detach().squeeze().numpy(), "Input Moving")
    utils.showNDA_InEditor_BW(fixed.detach().squeeze().numpy(), "Input Fixed")

    A = torch.autograd.Variable(torch.eye(3, dtype=torch.float32))

    dim = list(dim[2:])
    dim = tuple([1] + dim + [2])

    phi = torch.zeros(dim, dtype=torch.float32)

    for i in range(dim[-3]):
        for j in range(dim[-2]):
            point = torch.tensor([i, j, 1], dtype=torch.float32)
            normed = torch.matmul(normCoords, point)
            newPoint = torch.matmul(A, normed)
            normed = torch.divide(normed[:2], normed[2])
            newPoint = torch.divide(newPoint[:2], newPoint[2])
            traj = normed - newPoint  # THIS IS THE CRITICAL LINE WE'RE TESTING
            disp = normed + traj
            phi[0, i, j] = disp

    warped = F.grid_sample(moving, phi, mode='nearest')

    utils.showNDA_InEditor_BW(warped.detach().squeeze().numpy(), "Warped Result")

    print(F.cross_entropy(warped,fixed))

def grid_sample_v2():
    import torch
    import torch.nn.functional as F

    dim = [100, 100]

    normCoords = [[2.0 / dim[0], 0, -((dim[0] - 1) / dim[0])],
                  [0, 2.0 / dim[1], -((dim[0] - 1) / dim[0])],
                  [0, 0, 1]]

    # normCoords = np.linalg.inv(normCoords)

    normCoords = torch.tensor(normCoords, dtype=torch.float32)

    dim = [1, 1] + dim
    dim = tuple(dim)

    moving = torch.zeros(dim, dtype=torch.float32)
    fixed = torch.zeros(dim, dtype=torch.float32)

    print(moving.shape)
    for i in range(dim[-2]):
        for j in range(dim[-1]):
            if i > dim[-2] * .25 and i < dim[-2] * .75:
                if j > dim[-1] * .25 and j < dim[-1] * .75:
                    moving[0, 0, i, j] = 1.0

    for i in range(dim[-2]):
        for j in range(dim[-1]):
            if i > dim[-2] * .35 and i < dim[-2] * .85:
                if j > dim[-1] * .35 and j < dim[-1] * .85:
                    fixed[0, 0, i, j] = 1.0

    utils.showNDA_InEditor_BW(moving.detach().squeeze().numpy(), "Input Moving")
    utils.showNDA_InEditor_BW(fixed.detach().squeeze().numpy(), "Input Fixed")

    A = torch.autograd.Variable(torch.eye(3, dtype=torch.float32))

    dim = list(dim[2:])
    dim = tuple([1] + dim + [2])

    phi = torch.zeros(dim, dtype=torch.float32)

    for i in range(dim[-3]):
        for j in range(dim[-2]):
            point = torch.tensor([i, j, 1], dtype=torch.float32)
            normed = torch.matmul(normCoords, point)
            newPoint = torch.matmul(A, normed)
            normed = torch.divide(normed[:2], normed[2])
            newPoint = torch.divide(newPoint[:2], newPoint[2])
            traj = normed - newPoint  # THIS IS THE CRITICAL LINE WE'RE TESTING
            disp = normed + traj
            phi[0, i, j] = disp

    warped = F.grid_sample(moving, phi, mode='nearest')
    print(warped.shape)

    print(phi.max(), phi.min())

    utils.showNDA_InEditor_BW(warped.detach().squeeze().numpy(), "Warped Result")

def pytorch_v1():
    # External Modules
    import SimpleITK as sitk
    import numpy as np
    import torch
    from pytorch3d import transforms

    # Custom Classes
    import utilities as utils
    from src import Weights

    # STEP 1: Read in data
    movingData = sitk.ReadImage("../images/moving.nii")
    movingData = sitk.GetArrayFromImage(movingData)

    fixedData = sitk.ReadImage("../images/fixed.nii")
    fixedData = sitk.GetArrayFromImage(fixedData)

    componentSegmentations = {}
    numComponents = 8
    imageDimensions = fixedData.shape
    if len(imageDimensions) == 3:
        imageDepth = imageDimensions[0]
        imageWidth = imageDimensions[1]
        imageHeight = imageDimensions[2]
    else:
        imageDepth = 1
        imageWidth = imageDimensions[0]
        imageHeight = imageDimensions[1]

    for i in range(numComponents):
        temp = sitk.ReadImage("../images/segmentations/component" + str(i) + ".nii")
        componentSegmentations[i] = sitk.GetArrayFromImage(temp)

    print("Importing Image Data.")
    movingImage = torch.tensor(utils.normalizeImage(movingData), dtype=torch.float64)
    fixedImage = torch.tensor(utils.normalizeImage(fixedData), dtype=torch.float64)
    for idx, img in componentSegmentations.items():
        componentSegmentations[idx] = utils.normalizeImage(img)

    # utils.showNDA_InEditor_BW(movingData[10,:,:], "Moving Image")
    # utils.showNDA_InEditor_BW(fixedData[10,:,:], "Fixed Image")
    componentWeights = {}
    for idx, img in componentSegmentations.items():
        # utils.showNDA_InEditor_BW(img[10,:,:], "Component " + str(idx))
        componentWeights[idx] = 1 / numComponents  # assume for now that these are fixed gamma terms

    print("Normalizing Weights and Generating Weight Images.")
    # STEP 2: Calculate the Normalized Weight Volume
    weightImages = Weights.getNormalizedCommowickWeight(componentSegmentations, componentWeights)

    # for idx, img in weightImages.items():
    # utils.showNDA_InEditor_BW(img[10,:,:], "Weight Image for Component "+ str(idx))

    def _augmentDimensions(imageDimensions: tuple, augmentation):
        temp = list(imageDimensions)
        if type(augmentation) == int:
            temp.append(augmentation)
        elif type(augmentation) == list:
            temp = temp + augmentation
        else:
            aug = list(augmentation)
            temp = temp + aug
        return tuple(temp)

    dim = _augmentDimensions(imageDimensions, [numComponents])

    print("Composing Weight Volume")
    weightVolume = np.zeros(dim, dtype=np.float64)
    for idx in range(numComponents):
        weightImage = weightImages[idx]
        weightVolume[:, :, :, idx] = weightImage

    # utils.showNDA_InEditor_BW(weightImages[1][10,:,:],"Weight Image")
    # utils.showNDA_InEditor_BW(weightVolume[10,:,:,1], "Weight Volume Slice")

    weightVolume = torch.tensor(data=weightVolume, dtype=torch.float64, requires_grad=False)
    weightVolume = torch.reshape(weightVolume, shape=(imageWidth * imageHeight * imageDepth, numComponents))

    print("Initializing Component Transforms.")
    # STEP 3: Initialize transform variables
    eye = torch.zeros((4, 4), dtype=torch.float64)  # zeros for [ L v ] matrix
    eye = eye.reshape((1, 4, 4))
    componentTransforms = eye.repeat(numComponents, 1, 1)
    componentTransforms = torch.autograd.Variable(data=componentTransforms, requires_grad=True)
    # I stole the above pattern off the internet somewhere not sure entirely why it works
    # NB: Pytorch3D uses row-ordinal matrices of format (batch, row, col)
    # This means the affine matrices are reversed from their col-ordinal position!
    # [ R 0 ]
    # [ t 1 ]  If you put digits where the zero should be Pytorch3D throws an error.

    print("Entering Registration Loop.")
    # STEP 4: ENTER UPDATE LOOP
    stop_loss = 1e-5
    step_size = stop_loss / 3.0
    maxItrs = 1

    print(componentTransforms.shape)
    componentTransforms = torch.reshape(componentTransforms, (8, 16))
    print(weightVolume.shape)

    test = torch.matmul(weightVolume, componentTransforms)
    print(test.shape)

    for i in range(maxItrs):

        print("\tCalculating Logs Mappings and Fusing...")
        logMaps = transforms.se3_log_map(componentTransforms)
        # The se3 log map takes the form [ R 1 | t 1 ] a [1 ,6] row matrix
        # This function returns N = numComponents row matrices.

        fusedVectorLogs = torch.matmul(weightVolume, logMaps)

        LEPTImageDimensions = _augmentDimensions(imageDimensions, [4, 4])
        LEPTImageVolume = torch.zeros(LEPTImageDimensions, dtype=torch.float64)
        LEPTImageVolume = LEPTImageVolume.reshape((imageHeight * imageWidth * imageDepth, 4, 4))

        print("\tCalculating Exponential Mappings...")
        for i in range(imageWidth * imageHeight * imageDepth):
            LEPTImageVolume[i] = transforms.se3_exp_map(torch.reshape(fusedVectorLogs[i], (1, 6)))

        LEPTImageVolume = LEPTImageVolume.reshape(LEPTImageDimensions)

        print("\tCalculating Displacements...")
        displacementFieldDimensions = _augmentDimensions(imageDimensions, len(imageDimensions))
        displacementField = torch.zeros(displacementFieldDimensions, dtype=torch.float64)

        normCoords = [[2.0 / dim[0], 0, 0, -((dim[0] - 1) / dim[0])],
                      [0, 2.0 / dim[1], 0, -((dim[1] - 1) / dim[1])],
                      [0, 0, 2.0 / dim[2], -((dim[2] - 1) / dim[2])],
                      [0, 0, 0, 1]]
        norm = torch.tensor(normCoords, dtype=torch.float64)
        if (len(imageDimensions) == 3):
            for depth in range(imageDimensions[0]):
                print("Processing slice ", depth + 1, " of ", imageDepth, ".")
                for row in range(imageDimensions[1]):
                    for col in range(imageDimensions[2]):
                        homogeneousPoint = torch.tensor([depth, row, col, 1], dtype=torch.float64)
                        homogeneousPoint = torch.matmul(norm, homogeneousPoint)
                        newPoint = torch.matmul(LEPTImageVolume[depth, row, col], homogeneousPoint)
                        newPoint = torch.divide(newPoint, newPoint[len(imageDimensions)])[:len(imageDimensions)]
                        oldPoint = homogeneousPoint[:len(imageDimensions)]
                        displacementField[depth, row, col] = oldPoint - newPoint

    # STEP 5: Warp Image
    # Originally, we had discussed using SimpleITK.  There is an issue in that SimpleITK relies upon
    # numpy, and not pytorch, to calculate the displacement and loss.  This means that the gradient
    # propagation will be unhooked in the warping step and not able to propagate from the sitk loss
    # calculation back into the model.

    # TorchFields is a spacial warping package extending PyTorch for the express purpose of building
    # spatial transformer networks.  Unfortunately, it has not been actively maintained for over 3
    # years, and the feature set constrains itself to vector fields in two dimensions.  After examining
    # their code, it appears that their warp field function (.sample()) is a warper around another
    # pytorch function, grid_sample().  Grid sample is extensible to '5D' images (N,C,H,W,D) where
    # N is the batch, C is the channels, and height, width, depth is the same.

    displacementField = displacementField.unsqueeze(0)

    print(displacementField.max(), displacementField.min())

    import torch.nn.functional as F
    movingImage = movingImage.unsqueeze(0).unsqueeze(0)
    warped = F.grid_sample(movingImage, displacementField, mode='nearest', padding_mode='zeros', align_corners=False)
    utils.showNDA_InEditor_BW(warped.detach().squeeze().numpy()[10, :, :], "Result")
import copy

import numpy
import pytorch3d.transforms

import Polyrigid
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
    import torchvision as tv
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
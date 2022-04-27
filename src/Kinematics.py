import SimpleITK as sitk
import numpy as np
import os
import glob
import skimage.measure
from skimage.measure import label, regionprops

fpModel = "../images/results/" \
          "DICE_00_2500_MSE_00_2500_smooth_00_2500_nJD_00_0000_rigid_00_2500_trans_00_0000_weight_00_5000"

scaphoid = {'component':1}
lunate = {'component':2}
triquetral = {'component':3}
pisiform = {'component':4}
trapezoid = {'component':8}
trapezium = {'component':7}
capitate = {'component':6}
hamate = {'component':5}

carpalBones = [scaphoid,lunate,triquetral,pisiform,trapezoid,trapezium,capitate,hamate]

for file in glob.glob(os.path.join(fpModel,'frame_*_frame_*')):
    transforms = np.load(os.path.join(file,'component_transforms_final_euclidean.npz'))
    target = file.split('_')[-1]
    for i in range(0,8):
        carpalBones[i][target] = transforms[str(i)]

aCapitate = {'10':np.eye(4)}
for i in range(11,20):
    aCapitate[str(i)] = np.dot(capitate[str(i)],aCapitate[str(i-1)])
    print(aCapitate[str(i)])
for i in range(9,-1,-1):
    aCapitate[str(i)] = np.dot(capitate[str(i)],aCapitate[str(i+1)])
img = sitk.ReadImage(os.path.join(fpModel,"frame_10_to_frame_11/warped_seg_6.nii"))
npImg = sitk.GetArrayFromImage(img)
moments = skimage.measure.inertia_tensor(npImg)
centroid = skimage.measure.centroid(npImg)
lam, Q = np.linalg.eig(moments)
aCapitate['centroid'] = centroid
aCapitate['frame'] = Q
aCapitate['lengths'] = lam

aCarpalBones = [aCapitate]

img = sitk.ReadImage(os.path.join(fpModel,"frame_10_to_frame_11/warped_seg_6.nii"))
npImg = sitk.GetArrayFromImage(img)
moments = skimage.measure.inertia_tensor(npImg)
centroid = skimage.measure.centroid(npImg)
lam, Q = np.linalg.eig(moments)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def getTransform(bone: int,frame: int):
    homoCentroid = np.array([*aCarpalBones[bone]['centroid'], 1.])
    newCentroid = np.dot(aCarpalBones[bone][str(frame)], homoCentroid)
    newCentroid = (newCentroid / newCentroid[-1])[:-1]
    newRot = np.dot(aCarpalBones[bone][str(frame)][0:3, 0:3], aCarpalBones[bone]['frame'])
    return newCentroid, newRot

def update(frame):
    global quiver
    quiver.remove()
    for bone in range(len(aCarpalBones)):
        roid, rot = getTransform(bone,frame)
        quiver = ax.quiver(*roid, *rot[0], length=10, color='r')
        quiver = ax.quiver(*roid, *rot[1], length=10, color='b')
        quiver = ax.quiver(*roid, *rot[2], length=10, color='g')

fig, ax = plt.subplots(subplot_kw=dict(projection="3d",proj_type='ortho'))
a,b=getTransform(0,0)
quiver = ax.quiver(*a,*b)

for i in range(0,20):
    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_zlim(0,100)

    update(i)

    ax.azim = 70
    ax.elev = 10
    plt.show()

'''
def get_arrow(theta):
    x = np.cos(theta)
    y = np.sin(theta)
    z = 0
    u = np.sin(2*theta)
    v = np.sin(2*theta)
    w = np.cos(3*theta)
    return x,y,z,u,v,w

quiver = ax.quiver(*get_arrow(0))

#ax.set_xlim(-2,2)
#ax.set_ylim(-2,2)
#ax.set_zlim(-2,2)

#def update(theta):
#    global quiver
#    quiver.remove()
#    quiver = ax.quiver(*get_arrow(theta))

#ani = FuncAnimation(fig, update,frames=np.linspace(0,2*np.pi,200),interval=50)
plt.show()
'''
import SimpleITK as sitk
import numpy as np
import os
import glob
import skimage.measure
from skimage.measure import label, regionprops

if not os.path.exists('../results/'):
    os.mkdir('../results/')
if not os.path.exists('../results/plots/'):
    os.mkdir('../results/plots/')
if not os.path.exists('../results/plots/estimations/'):
    os.mkdir('../results/plots/estimations/')

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

aScaphoid = {'10':np.eye(4)}
aLunate = {'10':np.eye(4)}
aTriquetral = {'10':np.eye(4)}
aPisiform = {'10':np.eye(4)}
aTrapezium = {'10':np.eye(4)}
aTrapezoid = {'10':np.eye(4)}
aCapitate = {'10':np.eye(4)}
aHamate = {'10':np.eye(4)}
aCarpalBones = [aScaphoid,aLunate,aTriquetral,aPisiform,
                aTrapezium, aTrapezoid,aCapitate,aHamate]
for i in range(len(aCarpalBones)):
    aCarpalBones[i]['label'] = i
for bone in aCarpalBones:
    for i in range(11,20):
        bone[str(i)] = np.dot(carpalBones[bone['label']][str(i)],bone[str(i-1)])
    for i in range(9,-1,-1):
        bone[str(i)] = np.dot(carpalBones[bone['label']][str(i)],bone[str(i+1)])
    img = sitk.ReadImage(os.path.join(fpModel,"frame_10_to_frame_11/warped_seg_"
                                      + str(bone['label']+1)+".nii"))
    npImg = sitk.GetArrayFromImage(img)
    moments = skimage.measure.inertia_tensor(npImg)
    centroid = skimage.measure.centroid(npImg)
    lam, Q = np.linalg.eig(moments)
    bone['centroid'] = centroid
    bone['frame'] = Q.transpose()
    bone['lengths'] = lam

import matplotlib.pyplot as plt

def getTransform(bone: int,frame: int):
    homoCentroid = np.array([*aCarpalBones[bone]['centroid'], 1.])
    newCentroid = np.dot(aCarpalBones[bone][str(frame)], homoCentroid)
    newCentroid = (newCentroid / newCentroid[-1])[:-1]
    homoFrameX = np.array([*aCarpalBones[bone]['frame'][0],1.])
    homoFrameY = np.array([*aCarpalBones[bone]['frame'][1],1.])
    homoFrameZ = np.array([*aCarpalBones[bone]['frame'][2],1.])
    frameX = np.dot(aCarpalBones[bone][str(frame)],homoFrameX)
    frameY = np.dot(aCarpalBones[bone][str(frame)],homoFrameY)
    frameZ = np.dot(aCarpalBones[bone][str(frame)],homoFrameZ)
    newRotX = (frameX/frameX[-1])[:-1]
    newRotY= (frameY/frameY[-1])[:-1]
    newRotZ= (frameZ/frameZ[-1])[:-1]
    newRot = np.vstack((newRotX,newRotY,newRotZ))
    return newCentroid, newRot

def update(frame):
    global quiver
    quiver.remove()
    for bone in range(len(aCarpalBones)):
        roid, rot = getTransform(bone,frame)
        quiver = ax.quiver(*roid, *rot[0], length=10, color='r')
        quiver = ax.quiver(*roid, *rot[1], length=10, color='b')
        quiver = ax.quiver(*roid, *rot[2], length=10, color='g')
        ax.text(*roid,aCarpalBones[bone]['label'])

for i in range(0,20):
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw=dict(projection="3d",proj_type='ortho'))
    a,b=getTransform(0,0)
    quiver = ax.quiver(*a,*b)

    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_zlim(0,100)
    ax.azim = 180
    ax.elev = 90
    ax.invert_zaxis()
    ax.set_zticks([])
    fig.tight_layout()
    fig.suptitle("Estimated Carpal Bone \nTransformations",fontsize=24)
    ax.set_ylabel("Frame "+str(i))
    update(i)
    fig.savefig(f'../results/plots/estimations/frame_{i:02}')
    fig.clear()
    plt.close()

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
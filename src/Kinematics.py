import SimpleITK as sitk
import numpy as np
import torch
import os
import glob
import skimage.measure
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

vAlpha = 0.05
vElevation = 90
vAzim = 0

colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#e41a1c', '#dede00']

comps = ['Scapohid','Lunate','Triquetral','Pisiform',
         'Hamate','Capitate','Trapezoid','Trapezium']

# Create a list of coordinates
Pd = torch.range(0,59)
Ph = torch.range(0,99)
Pw = torch.range(0,99)
P = torch.cartesian_prod(Pd,Ph,Pw)
P = P.reshape(60,100,100,3).detach().numpy()

# Set up plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=vElevation, azim=vAzim)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 60)
ax.set_zticks([])
fig.suptitle("Estimated Carpal Bone Positions", fontsize=24)
ax.set_title("Frame 0", fontsize=14)
fig.tight_layout()

for comp in range(1,9):
    compColor = comp-1
    compName = comps[comp-1]

    # Read in the component warped label image
    img = sitk.ReadImage("/home/russ/github/polyrigid/images/results"
                          "/DICE_00_0000_MSE_00_4878_smooth_00_0122_nJD_00_0122_rigid_00_2439_trans_00_2439_weight_00_5000/"
                          "frame_1_to_frame_0/warped_seg_"+str(comp)+".nii")
    npImg = sitk.GetArrayFromImage(img)

    # Plot the component point cloud
    ptCloud = np.where(np.stack((npImg,)*3,axis=-1) >= 0.7, P,0)
    ptCloud = np.ma.masked_equal(ptCloud,0)
    ax.scatter(ptCloud[:,:,:,1].flatten(),
               ptCloud[:,:,:,2].flatten(),
               ptCloud[:,:,:,0].flatten(),
               color=colors[compColor],
               alpha=vAlpha)

    centroid = skimage.measure.centroid(npImg)
    moments = skimage.measure.inertia_tensor(npImg)
    lam, Q = np.linalg.eig(moments)
    frame1 = Q[:,0]/np.linalg.norm(Q[:,0])
    frame2 = Q[:,1]/np.linalg.norm(Q[:,1])
    frame3 = Q[:,2]/np.linalg.norm(Q[:,2])
    ax.scatter(centroid[1],centroid[2],centroid[0],
               s=100,color=colors[compColor])

    ax.quiver(centroid[1],centroid[2],centroid[0],*frame1,color='r',length=lam[0]*0.2)
    ax.quiver(centroid[1],centroid[2],centroid[0],*frame2,color='g',length=lam[1]*0.2)
    ax.quiver(centroid[1],centroid[2],centroid[0],*frame3,color='b',length=lam[2]*0.2)

plt.show()



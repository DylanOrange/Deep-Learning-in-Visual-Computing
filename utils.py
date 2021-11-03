import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tools.fusion import rigid_transform

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def visualization(cube,alpha):
    # cube :(3,cube_D,cube_D,cube_D)  alpha: transparency
    # visualization of colored cube
    cube_D = cube.shape[-1]
    spatial_axes = [cube_D, cube_D, cube_D]
    filled = np.ones(spatial_axes, dtype=bool)
    alpha = 0.7
    colors = np.empty(spatial_axes + [4], dtype=np.float32)
    cube = cube.reshape(64,64,64,3)
    colors[:, :, :, :-1] = cube
    colors[:, :, :, -1] = alpha
    #fig = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(filled, facecolors=colors, edgecolors='k')
    plt.show()
def vox2world(vol_origin, vox_coords, vox_size):  # convert voxel grid coordinates to world coordinates
    # vol_origin: (3,)
    # vol_coords:(N,3)
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    for i in range(vox_coords.shape[0]):
        for j in range(3):
            cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
    return cam_pts

def cam2pix(cam_pts, intrmat):  # convert camera coordiantes to pixel coordiantes
    # cam_pts:(N,3)
    #print('shape of cam_pts{}'.format(cam_pts.shape))
    intrmat = intrmat.astype(np.float32)  # intr [ fx 0 ox;0 fy oy; 0 0 1]
    fx, fy = intrmat[0, 0], intrmat[1, 1],
    ox, oy = intrmat[0, 2], intrmat[1, 2],
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)  # (N,2)
    #print("shape of pix:{}".format(pix.shape))
    for i in range(cam_pts.shape[0]):
        pix[i, 0] = int((np.round(cam_pts[i, 0] * fx) / cam_pts[i, 2]) + ox)  # u = fx*X/Z + ox
        pix[i, 1] = int((np.round(cam_pts[i, 1] * fx) / cam_pts[i, 2]) + oy)  # v = fy*Y/Z + oy
    return pix

def get_CVC(color_im,cam_intr,cam_pose,vol_bonds, voxel_size,voxel_dim = 64):
    # initialization of cube
    vol_dim = np.array([3,voxel_dim,voxel_dim,voxel_dim])
    vol_origin =vol_bonds[:,0]

    # Initialize tsdf_vol and weight_vol
    cvc = np.zeros((vol_dim))  #(3,64,64,64)

    # get voxel grid coordiantes
    xv, yv, zv = np.meshgrid(range(vol_dim[1]),
                             range(vol_dim[2]),
                             range(vol_dim[3]),
                             indexing='ij')
    vox_coords = np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0).astype(int)
    vox_coords = vox_coords.T  #(N,3)
    #print("shape of vox_coords:{}".format(vox_coords.shape))
    im_h, im_w ,_= color_im.shape

    cam_pts = vox2world(vol_origin, vox_coords,voxel_size)  # coordinates of voxel in World coordinate (N,3)
    cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))  # world coordiante -> camera coordiante
    pix_z = cam_pts[:, 2]
    pix = cam2pix(cam_pts, cam_intr)
    #print("shape of pix:{}".format(pix.shape))
    pix_x, pix_y = pix[:, 0], pix[:, 1]  #(N,) (N,)

    # eliminate pixels outside view frustum
    valid_pix = np.logical_and(pix_x >= 0,
                               np.logical_and(pix_x < im_w,
                                              np.logical_and(pix_y > 0,
                                                             np.logical_and(pix_y < im_h,
                                                                            pix_z > 0))))

    rgb_val = np.zeros((len(pix_x),3))  #(N,3)
    rgb_val[valid_pix] = color_im[pix_y[valid_pix], pix_x[valid_pix]]
    rgb_val_valid = rgb_val[valid_pix]
    #print("shape of rgb_val_valid:{}".format(rgb_val_valid.shape))
    valid_vox_x = vox_coords[
        valid_pix, 0]  # vox_coords have already been considered voxel_size, (N,),(N,),(N,)
    valid_vox_y = vox_coords[
        valid_pix, 1]  # vox_coords have already been considered voxel_size, (N,),(N,),(N,)
    valid_vox_z = vox_coords[
        valid_pix, 2]  # vox_coords have already been considered voxel_size, (N,),(N,),(N,)
    cvc[:,valid_vox_x,valid_vox_y,valid_vox_z] = rgb_val_valid.reshape(3,-1)
    return cvc



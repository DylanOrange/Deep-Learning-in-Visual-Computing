import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import utils
import os
import pickle
import cv2
from utils import *
class ScanNet(Dataset):
    def __init__(self, scene, data_path, max_depth):
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        self.n_imgs = len(sorted(os.listdir(os.path.join(self.data_path, 'scans', scene, 'color'))))
        self.id_list = [i for i in range(self.n_imgs)]
        self.intr = np.loadtxt(os.path.join(self.data_path, 'scans', scene, 'intrinsic', 'intrinsic_color.txt'),
                               delimiter=' ')[:3, :3]
        with open(os.path.join(self.data_path, 'tsdf', scene, 'tsdf_info.pkl'), 'rb') as f:
            self.tsdf_info = pickle.load(f)
        self.voxel_size = self.tsdf_info['voxel_size']
        self.vol_bounds = self.tsdf_info['vol_bounds']
        self.gt = np.load(os.path.join(data_path,'tsdf',scene,'occ.npz'))['arr_0']

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, index):
        # print(index)
        index = self.id_list[index]
        # get camera pose
        cam_pose = np.loadtxt(os.path.join(self.data_path,'scans', self.scene, 'pose', str(index) + '.txt'), delimiter=' ')
        # assert cam_pose.shape == (4, 4)

        # get depth image
        depth_image = cv2.imread(os.path.join(self.data_path, 'scans',self.scene, 'depth', str(index) + '.png'), -1).astype(
            np.float32)
        depth_image /= 1000.
        depth_image[depth_image > self.max_depth] = 0

        # get rgb image
        color_image = cv2.imread(os.path.join(self.data_path, 'scans',self.scene, 'color', str(index) + '.jpg'))
        color_image = cv2.cvtColor((color_image), cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]),
                                 interpolation=cv2.INTER_AREA)  # a little confused about the size --> it's right

        # assert color_image.shape[:-1] == depth_image.shape and color_image.shape[-1] == 3
        cvc = get_CVC(color_image,self.intr,cam_pose,self.vol_bounds,self.voxel_size,voxel_dim=64)
        #return cam_pose, depth_image, color_image
        return cvc

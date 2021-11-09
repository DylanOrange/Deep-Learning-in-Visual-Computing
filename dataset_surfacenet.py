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
    def __init__(self, scene, data_path, max_depth,train=True):
        self.train = train
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        if self.train:
            self.n_imgs = int(len(sorted(os.listdir(os.path.join(self.data_path, 'scans', scene, 'color'))))*0.8)
        else:
            self.n_imgs =int(len(sorted(os.listdir(os.path.join(self.data_path, 'scans', scene, 'color'))))*0.2)
        self.id_list = [i for i in range(self.n_imgs)]
        self.intr = torch.tensor(np.loadtxt(os.path.join(self.data_path, 'scans', scene, 'intrinsic', 'intrinsic_color.txt'),
                               delimiter=' ')[:3, :3]).to(self.device)
    def __len__(self):
        return int(self.n_imgs/10)
    #  directly get CVC
    # def __getitem__(self, index):
    #     # print(index)
    #     index = self.id_list[index]
    #     # get camera pose
    #     cam_pose = np.loadtxt(os.path.join(self.data_path,'scans', self.scene, 'pose', str(index) + '.txt'), delimiter=' ')
    #     # assert cam_pose.shape == (4, 4)
    #
    #     # get depth image
    #     depth_image = cv2.imread(os.path.join(self.data_path, 'scans',self.scene, 'depth', str(index) + '.png'), -1).astype(
    #         np.float32)
    #     depth_image /= 1000.
    #     depth_image[depth_image > self.max_depth] = 0
    #
    #     # get rgb image
    #     color_image = cv2.imread(os.path.join(self.data_path, 'scans',self.scene, 'color', str(index) + '.jpg'))
    #     color_image = cv2.cvtColor((color_image), cv2.COLOR_BGR2RGB)
    #     color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]),
    #                              interpolation=cv2.INTER_AREA)  # a little confused about the size --> it's right
    #
    #     # assert color_image.shape[:-1] == depth_image.shape and color_image.shape[-1] == 3
    #     cvc = get_CVC(color_image,self.intr,cam_pose,self.vol_bounds,self.voxel_size,voxel_dim=64)
    #     #return cam_pose, depth_image, color_image
    #     return cvc

    # get 10 images at one time
    def __getitem__(self, index):
        #print('start getitem')
        if self.train:
            start = 10 * index
            end = start+10
            index_list = [x for x in range(start,end)]
        else:
            start = 10*index +1600
            end = start +10
            index_list = [x for x in range(start,end)]
        # get camera pose
        # assert cam_pose.shape == (4, 4)
        with open(os.path.join(self.data_path, 'tsdf', self.scene, 'tsdf_info_'+str(end)+'.pkl'), 'rb') as f:
            tsdf_info = pickle.load(f)
        voxel_size = torch.tensor(tsdf_info['voxel_size']).to(self.device)
        vol_bounds = torch.tensor(tsdf_info['vol_bounds']).to(self.device)
        gt = np.load(os.path.join(self.data_path,'tsdf',self.scene,'occ_'+str(end)+'.npz'))['arr_0']
        gt = torch.tensor(gt).to(self.device)
        gt = gt.unsqueeze(0)
        # get depth image
        color = []
        cam_pose_list = []
        for i in index_list:
            #depth_image = cv2.imread(os.path.join(self.data_path, 'scans',self.scene, 'depth', str(i) + '.png'), -1).astype(
                #np.float32)
            # depth_image /= 1000.
            # depth_image[depth_image > self.max_depth] = 0

            # get rgb image
            color_image = cv2.imread(os.path.join(self.data_path, 'scans',self.scene, 'color', str(i) + '.jpg'))
            color_image = cv2.cvtColor((color_image), cv2.COLOR_BGR2RGB)
            color_image = cv2.resize(color_image, (640, 480),
                                     interpolation=cv2.INTER_AREA)  # a little confused about the size --> it's right
            color_image = torch.tensor(color_image).to(self.device)
            color_image = color_image.unsqueeze(0)
            color.append(color_image)
            cam_pose = np.loadtxt(os.path.join(self.data_path, 'scans', self.scene, 'pose', str(i) + '.txt'),
                                  delimiter=' ')
            cam_pose = torch.tensor(cam_pose).to(self.device)
            cam_pose_list.append(cam_pose)
        color_image = torch.cat(color,axis=0)
        info = {'intr':self.intr,
                'cam_pose':cam_pose_list,
                'vol_bonds':vol_bounds,
                'voxel_size':voxel_size}
        # assert color_image.shape[:-1] == depth_image.shape and color_image.shape[-1] == 3
        #cvc = get_CVC(color_image,self.intr,cam_pose,self.vol_bounds,self.voxel_size,voxel_dim=64)
        #return cam_pose, depth_image, color_image
        #print('finish getitem')
        return color_image, info, gt

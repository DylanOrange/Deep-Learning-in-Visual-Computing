import numpy as np
import open3d as o3d
import os
import os
from torch.utils.data import Dataset
import cv2


def collate_fn(list_data):
    cam_pose, depth_im, _ = list_data
    # Concatenate all lists
    return cam_pose, depth_im, _


class Preparation(Dataset):
    def __init__(self, n_imgs, scene, data_path, max_depth):
        self.n_imgs = n_imgs
        print("total number of images:{}".format(self.n_imgs))
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        self.id_list = [i for i in range(n_imgs)]

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, index):
        #print(index)
        index = self.id_list[index]
        # get camera pose
        cam_pose = np.loadtxt(os.path.join(self.data_path, self.scene, 'pose', str(index) + '.txt'), delimiter=' ')
        assert cam_pose.shape == (4, 4)

        # get depth image
        depth_image = cv2.imread(os.path.join(self.data_path, self.scene, 'depth', str(index) + '.png'), -1).astype(
            np.float32)
        depth_image /= 1000.
        depth_image[depth_image > self.max_depth] = 0

        # get rgb image
        color_image = cv2.imread(os.path.join(self.data_path, self.scene, 'color', str(index) + '.jpg'))
        color_image = cv2.cvtColor((color_image),cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]),
                                 interpolation=cv2.INTER_AREA)  # a little confused about the size --> it's right

        assert color_image.shape[:-1] == depth_image.shape and color_image.shape[-1] == 3

        return cam_pose, depth_image, color_image

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import utils


class DTUdataset(Dataset):
    def __init__(self, img_dir, pos_dir, target_dir, bbox_dir, resol, size):
        self.img_dir = img_dir
        self.pos_dir = pos_dir
        self.target_dir = target_dir
        self.bbox_dir = bbox_dir
        self.imgs = os.listdir(img_dir)
        self.poses = os.listdir(pos_dir)
        self.bboxs = os.listdir(bbox_dir)
        self.resol = resol
        self.size = size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # TODO: add gt value
        img_path = os.path.join(self.image_dir, self.imgs[index])
        pos_path = os.path.join(self.pos_dir, self.poses[index])
        bbox_path = os.path.join(self.bbox, self.poses[index])
        image = np.array(Image.open(img_path)) / 255
        position = np.loadtxt(pos_path)
        bbox = np.loadtxt(bbox_path)  # np.array,  (2,3)
        colored_cube = utils.gen_colored_cubes(position, image, (bbox[0, 0], bbox[0, 1], bbox[0, 2]), self.resol,
                                               self.size)

        return colored_cube

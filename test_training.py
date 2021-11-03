import os
import sys

import numpy as np

sys.path.append('.')

from tools.fusion  import *
import pickle
import argparse
from tqdm import tqdm
from dataset import *
from torch.utils.data import DataLoader
from utils import *
import torch.optim as optim
from model import SurfaceNet
import matplotlib.pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser(description='Generation of Ground Truth')
    parser.add_argument("--data_path", metavar="DIR", help='path to raw data', default='/Users/zhengzhisheng/PycharmProjects/Deep-Learning-in-Visual-Computing/scannet')
    parser.add_argument("--save_name", metavar="DIR", help="file name", default="tsdf")
    parser.add_argument("--max_depth", default=3, type=int)
    parser.add_argument("--margin", default=3, type=int)
    parser.add_argument("--voxel_size", default=0.04, type=float)
    parser.add_argument("--window_size", default=9, type=int)
    parser.add_argument("--min_angle", default=15, type=float)
    parser.add_argument("--min_distance", default=0.1, type=float)
    return parser.parse_args()


args = parse_args()
epochs = 120

if __name__ == '__main__':
    model = SurfaceNet(30,1)
    optimizer = optim.Adam(model.parameters(),lr=0.005)
    dataset = ScanNet('scene0000_00',args.data_path,args.max_depth)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None,batch_sampler=None)
    loss_list=[]
    loss_fn = torch.nn.BCELoss()
    print("concatenating...")
    for id, cvc in enumerate(dataloader):
        if id ==0:
            CVC = cvc
        else:
            CVC = torch.cat((CVC,cvc),axis=0)
        print("concatenating: {}/{}".format(id+1,10))
    CVC = CVC.unsqueeze(0)
    CVC = CVC.float()
    gt = torch.tensor(dataset.gt)
    gt = gt.unsqueeze(0)
    gt = gt.unsqueeze(0)
    gt = gt.float()
    print("training start")
    for epoch in range(epochs):
        pred = model(CVC)
        loss = loss_fn(pred,gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        print("{}/{}, loss: {}".format(epoch,epochs,loss.item()))
    plt.plot(loss_list)
    plt.title("training loss -- one example")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    result = model(CVC)
    result = result.detach().numpy()
    np.savez_compressed(os.path.join(args.data_path, 'result','trainig_result'), result)


import os
import sys
import numpy as np
import torch.cuda
sys.path.append('.')
from tools.fusion  import *
import pickle
import argparse
from tqdm import tqdm
from dataset_surfacenet import *
from torch.utils.data import DataLoader
from utils import *
import torch.optim as optim
from model_surfacenet import SurfaceNet
import matplotlib.pyplot as plt
import tensorboard
from torch.utils.tensorboard import SummaryWriter
def parse_args():
    parser = argparse.ArgumentParser(description='Generation of Ground Truth')
    parser.add_argument("--data_path", metavar="DIR", help='path to raw data', default='./scannet')
    parser.add_argument("--save_name", metavar="DIR", help="file name", default="tsdf")
    parser.add_argument("--max_depth", default=3, type=int)
    parser.add_argument("--margin", default=3, type=int)
    parser.add_argument("--voxel_size", default=0.04, type=float)
    parser.add_argument("--window_size", default=9, type=int)
    parser.add_argument("--min_angle", default=15, type=float)
    parser.add_argument("--min_distance", default=0.1, type=float)
    return parser.parse_args()
args = parse_args()
# Hyperparameters
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
NUM_EPOCHS = 1
NUM_WORKERS = 0
TRAIN_IMG_DIR = ''
TRAIN_POS_DIR = ''
TRAIN_BBOX_DIR = ''
VAL_IMG_DIR = ''
VAL_POS_DIR = ''
VAL_BBOX_DIR = ''
LOAD_MODEL = False
if __name__ == '__main__':
    writer = SummaryWriter('./runs')
    model = SurfaceNet(30,1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(),lr=0.005)
    dataset_train = ScanNet('scene0000_00',args.data_path,args.max_depth,train=True)
    dataset_val = ScanNet('scene0000_00',args.data_path,args.max_depth,train=False)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1)
    loss_fn = torch.nn.BCELoss()
    print('start training')
    for epoch in range(NUM_EPOCHS):
        for i, (color_image,info,gt) in enumerate(dataloader_train):
            #print('forward!')
            pred = model(color_image,info)
            optimizer.zero_grad()
            loss = loss_fn(pred,gt)
            loss.backward()
            optimizer.step()
            #print('finish backward!')
            writer.add_scalar('Training loss',
                              loss.item(),
                              epoch * len(dataloader_train) + i)
        # see validation loss
            model.eval()
            random = np.random.randint(0,40)
            color_image,info,gt = dataset_val[random]
            pred = model(color_image,info)
            gt = gt.unsqueeze(0)
            with torch.no_grad():
                val_loss = loss_fn(pred,gt)
            writer.add_scalar('Validation loss',
                                  val_loss.item(),
                                  epoch * len(dataloader_train) + i)
            
            model.train()
            
            
            print('epoch:{}/{}, iteration:{}/{}, loss:{}, val_loss:{}'.format(epoch,NUM_EPOCHS,i,len(dataloader_train),loss.item(),val_loss.item()))
    print('end training')






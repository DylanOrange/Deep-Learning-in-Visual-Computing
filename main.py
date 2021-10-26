import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import SurfaceNet
from dataset import DTUdataset
from torch.utils.data import DataLoader
from utils import save_checkpoint, load_checkpoint

# Hyperparameters
LEARNING_RATE = 1E-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 0
TRAIN_IMG_DIR = ''
TRAIN_POS_DIR = ''
TRAIN_BBOX_DIR = ''
VAL_IMG_DIR = ''
VAL_POS_DIR = ''
VAL_BBOX_DIR = ''
LOAD_MODEL = False
# training
train_dataset = DataLoader(DTUdataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
model = SurfaceNet().to(device=DEVICE)
loop = tqdm(train_dataset)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)
loss=[]
for epoch in range(NUM_EPOCHS):
    temp_loss=[]
    for batch, (colored_cube, targets) in enumerate(loop):
        # forward
        colored_cube = colored_cube.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        prediction = model(colored_cube)
        loss = loss_fn(prediction, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        temp_loss.append(loss.item())
    avg_loss = np.average(temp_loss)
    loss.append(avg_loss)
    checkpoint = {
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict()
    }
    filename = 'checkpoint_' + str(epoch) +'_.pth.tar'
    save_checkpoint(checkpoint,filename=filename)

    if (epoch+1)%20 == 0:
        print("epoch: {} , avg loss: {} ".format(epoch,np.average(loss)))

print("Finished Training")




import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DilConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=2, dilation=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=2, dilation=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=2, dilation=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SurfaceNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 80, 160, 300]):
        super(SurfaceNet, self).__init__()
        self.l1 = TripleConv(in_channels, features[0])
        self.l2 = TripleConv(features[0],features[1])
        self.l3 = TripleConv(features[1],features[2])
        self.l4 = DilConv(features[2],features[3])
        self.l5 = TripleConv(16*4,100)
        self.y = nn.Conv3d(100,out_channels,1,padding=0,stride=1)
        self.pool = nn.MaxPool3d(2, stride=2)  # s -> s/2
        self.upconv1 = nn.ConvTranspose3d(features[0], 16, 1, stride=2, padding=0, output_padding=1) #double size
        self.upconv2 = nn.ConvTranspose3d(features[1], 16, 1, stride=4, padding=0, output_padding=3) #four times size
        self.upconv3 = nn.ConvTranspose3d(features[2], 16, 1, stride=4, padding=0, output_padding=3) #four times size
        self.upconv4 = nn.ConvTranspose3d(features[3], 16, 1, stride=4, padding=0, output_padding=3) #four times size
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x1 = self.l1(x)  #s ->s
        x1 = self.pool(x1) # s->s/2
        s1 = self.upconv1(x1) # s/2 -> s
        #print("shape of x:{},shape of s:{}".format(x1.shape,s1.shape))

        x2 = self.l2(x1)   #s/2 -> s/2
        x2 = self.pool(x2)  #s/2 -> s/4
        s2 = self.upconv2(x2) #s/4 -> s
        #print("shape of x:{},shape of s:{}".format(x2.shape, s2.shape))
        x3 = self.l3(x2)   #s/4 -> s/4
        s3 = self.upconv3(x3) #s/4 -> s
        #print("shape of x:{},shape of s:{}".format(x3.shape, s3.shape))
        x4 = self.l4(x3)   #s/4 -> s/4
        s4 = self.upconv4(x4)  #s/4 ->s
        #print("shape of x:{},shape of s:{}".format(x4.shape, s4.shape))
        s5 = torch.cat([s1,s2,s3,s4],dim=1) #s->s
        #print("shape of s5:{}".format(s5.shape))
        s5 = self.l5(s5)
        output = self.y(s5)
        output = self.sigmoid(output)
        return output

if __name__ == '__main__':
    x = torch.rand(1,30, 64, 64, 64)
    testnet = SurfaceNet(30,1)
    x = testnet(x)
    print(x.shape)

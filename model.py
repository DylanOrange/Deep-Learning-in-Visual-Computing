import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from backbone import MnasMulti
from torch.nn.functional import grid_sample


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
        self.feature_extraction = MnasMulti()

    def forward(self,x,info):
        cvc_list =[]
        x = x.squeeze(0)  
        for i in range(10):
            image = x[i,:].permute(2,0,1).contiguous() #480,640,3 -> 3,480,640
            image = image.unsqueeze(0)  #3,480,640 ->1,3,480,640
            features = (self.feature_extraction(image.float()))[1] #1,40,60,80
            volume = CVC(info["vol_bonds"],info['voxel_size'])
            volume.backproject_features(features,info['intr'],info['cam_pose'][i])
            cvc = volume.get_cvc()
            cvc_list.append(cvc)
        cvc_integrated = torch.cat(cvc_list,axis=0)
        cvc_integrated = cvc_integrated.unsqueeze(0)#1,30,64,64,64
        x1 = self.l1(cvc_integrated)  #s ->s
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





class CVC:
    def __init__(self, vol_bonds, voxel_size, margin=5):
        """Constructor.

        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Define voxel volume parameters
        #print(vol_bonds.shape)
        self._vol_bonds = vol_bonds.squeeze(0)
        self._voxel_size = voxel_size
        self._sdf_trunc = margin * self._voxel_size
        self._const = 256 * 256
        self._integrate_func = integrate
        self._backproject_features = backproject_features

        # Adjust volume bounds
        self._vol_dim = torch.tensor([64, 64, 64]).long()
        self._vol_origin = self._vol_bonds[:, 0]

        # Get voxel grid coordinates
        xv, yv, zv = torch.meshgrid(
            torch.arange(0, self._vol_dim[0]),
            torch.arange(0, self._vol_dim[1]),
            torch.arange(0, self._vol_dim[2]),
        )
        self._vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(self.device)

        # Convert voxel coordinates to world coordinates
        self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords)
        self._world_c = torch.cat([
            self._world_c, torch.ones(len(self._world_c), 1, device=self.device)], dim=1)

        self.reset()

        # print("[*] voxel volume: {} x {} x {}".format(*self._vol_dim))
        # print("[*] num voxels: {:,}".format(self._num_voxels))

    def reset(self):
        self.cvc = torch.ones(41,64,64,64).to(self.device)
        # self._weight_vol = torch.zeros(*self._vol_dim).to(self.device)
        # self._color_vol = torch.zeros(*self._vol_dim).to(self.device)
        # self._occ_vol = torch.zeros(*self._vol_dim).to(self.device)

    def integrate(self, color_im, cam_intr, cam_pose):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign to the current observation.
        """
        cam_pose = cam_pose.clone().detach().float().to(self.device)
        cam_intr = cam_intr.clone().detach().float().to(self.device)
        color_im = color_im.clone().detach().float().to(self.device)
        im_h, im_w,_ = color_im.shape
        cvc= self._integrate_func(
            cam_intr,
            cam_pose,
            color_im,
            self._world_c,
            self._vox_coords,
            self.cvc,
            im_h, im_w,
        )
        self.cvc = cvc

    def backproject_features(self, features, cam_intr, cam_pose): 

        cam_pose = cam_pose.clone().detach().float().to(self.device)
        cam_intr = cam_intr.clone().detach().float().to(self.device)
        features = features.clone().detach().float().to(self.device)
        cvc = self._backproject_features(cam_intr,
                cam_pose,
                features,
                self._world_c,
                self._vox_coords,
                self.cvc)
        self.cvc = cvc


    def get_cvc(self):
        return self.cvc

    @property
    def sdf_trunc(self):
        return self._sdf_trunc

    @property
    def voxel_size(self):
        return self._voxel_size
def integrate(
        cam_intr,
        cam_pose,
        color_im,
        world_c,
        vox_coords,
        cvc,
        im_h,
        im_w,
):
    # Convert world coordinates to camera coordinates
    cam_pose = cam_pose.squeeze(0)
    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam.float(), world_c.float().transpose(1, 0)).transpose(1, 0).float()
    cam_intr = cam_intr.squeeze(0)
    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
    pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

    # Eliminate pixels outside view frustum
    valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
    rgb_val = torch.zeros(len(pix_x),3).to('cuda')
    rgb_val[valid_pix] = color_im[pix_y[valid_pix], pix_x[valid_pix]]
    rgb_val_valid = rgb_val[valid_pix]
    # assign rgb value
    valid_vox_x = vox_coords[valid_pix, 0]
    valid_vox_y = vox_coords[valid_pix, 1]
    valid_vox_z = vox_coords[valid_pix, 2]
    cvc[:,valid_vox_x,valid_vox_y,valid_vox_z] = rgb_val_valid.reshape(3,-1)
    return cvc

def backproject_features(
    cam_intr,
    cam_pose,
    features,
    world_c,
    vox_coords,
    cvc):
    _,c,im_h,im_w = features.shape
    # Convert world coordinates to camera coordinates
    cam_pose = cam_pose.squeeze(0)
    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam.float(), world_c.float().transpose(1, 0)).transpose(1, 0).float()
    cam_intr = cam_intr.squeeze(0)
    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
    pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

    pixel_grid = torch.stack([2 * pix_x / (im_w - 1) - 1, 2 * pix_y / (im_h - 1) - 1], dim=-1)

    mask = pixel_grid.abs() <= 1
    mask = (mask.sum(dim=-1) == 2) & (pix_z > 0)

    pixel_grid = pixel_grid.view(1, 1, -1, 2)
    features = grid_sample(features, pixel_grid, padding_mode='zeros', align_corners=True)#1,40,60,80->1,40,1,64*64*64 
    
    features = features.view(1, c, -1)  #10，80，13824
    mask = mask.view(1, -1)
    pix_z = pix_z.view(1, -1)

    # remove nan
    features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
    pix_z[mask == False] = 0

    features = features.sum(dim=0)
    mask = mask.sum(dim=0)
    invalid_mask = mask == 0
    mask[invalid_mask] = 1
    in_scope_mask = mask.unsqueeze(0)
    features /= in_scope_mask
    features = features.permute(1, 0).contiguous()

    pix_z = pix_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
    im_z_mean = pix_z[pix_z > 0].mean()
    im_z_std = torch.norm(pix_z[pix_z > 0] - im_z_mean) + 1e-5
    im_z_norm = (pix_z - im_z_mean) / im_z_std
    im_z_norm[pix_z <= 0] = 0
    features = torch.cat([features, im_z_norm], dim=1) #64*64*64,41

    # Eliminate pixels outside view frustum
    # valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
    rgb_val = torch.zeros(len(pix_x),c+1).to('cuda')
    rgb_val_valid = features[mask]
    # assign rgb value
    valid_vox_x = vox_coords[mask, 0]
    valid_vox_y = vox_coords[mask, 1]
    valid_vox_z = vox_coords[mask, 2]
    cvc[:,valid_vox_x,valid_vox_y,valid_vox_z] = rgb_val_valid.reshape(c+1,-1)
    return cvc

if __name__ == '__main__':
    x = torch.rand(1,30, 64, 64, 64)
    testnet = SurfaceNet(30,1)
    x = testnet(x)
    print(x.shape)

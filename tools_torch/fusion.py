import torch
import numpy as np
from skimage import measure


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])  # add one more column  (5,3) --> (5,4)
    xyz_t_h = np.dot(transform, xyz_h.T).T  # (4,4) * (4,5)  ---> (4,5) ---> (5,4)
    return xyz_t_h[:, :3]


class tsdfvolume:

    def __init__(self, vol_bonds, voxel_size, margin=3):
        self.vol_bonds = np.asarray(vol_bonds)
        #print("vol_bonds:{}".format(vol_bonds))
        assert vol_bonds.shape == (3, 2)  # xim,xmax, ymin,ymax, zmin,zmax
        self.voxel_size = voxel_size
        self.voxel_size = np.max(self.vol_bonds[:,1]-self.vol_bonds[:,0])/64  # reset voxel size to get a 64*64*64 cube
        self.trunc_margin = margin * self.voxel_size
        self._color_const = 256 * 256

        # Adjust volume bounds

        # self.vol_dim = np.round((self.vol_bonds[:, 1] - self.vol_bonds[:, 0]) / self.voxel_size).copy(order='C').astype(int)
        # # (3,)  number of voxels in each axis
        self.vol_dim = np.array([64,64,64])
        self.vol_bonds[:,1] = self.vol_bonds[:,0] + self.vol_dim * self.voxel_size
        self.vol_origin = self.vol_bonds[:, 0].copy(order='C').astype(np.float32)

        # Initialize tsdf_vol and weight_vol
        self.tsdf_vol = np.ones(self.vol_dim).astype(np.float32)
        self.weight_vol = np.zeros(self.vol_dim).astype(np.float32)
        self.color_vol = np.zeros(self.vol_dim).astype(np.float32)

        # get voxel grid coordiantes
        xv, yv, zv = np.meshgrid(range(self.vol_dim[0]),
                                 range(self.vol_dim[1]),
                                 range(self.vol_dim[2]),
                                 indexing='ij')
        self.vox_coords = np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0).astype(int)
        self.vox_coords = self.vox_coords.T
        print("the shape of vox_coords:{}".format(self.vox_coords.shape))

    def vox2world(self,vol_origin, vox_coords, vox_size):  # convert voxel grid coordinates to world coordinates
        # vol_origin: (3,)
        # vol_coords:(N,3)
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in range(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    def cam2pix(self,cam_pts, intrmat):  # convert camera coordiantes to pixel coordiantes
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

    def update_tsdf(self, tsdf_vol, dist, w_old, obs_weight=1):
        tsdf_new = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in range(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_new[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_new, w_new

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1):
        #   implement KinectFusion
        im_h, im_w = depth_im.shape
        if color_im is not None:
            # Fold RGB color image into a single channel image
            color_im = color_im.astype(np.float32)
            color_im = np.floor(color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[
                ..., 0])  # color_const = 256*256
            #print("the shape of color_im:{}".format(color_im.shape))
            #color_im = color_im.reshape(-1).astype(np.float32)  # faltten  (HxW,)
        else:
            color_im = np.array(0)

        cam_pts = self.vox2world(self.vol_origin, self.vox_coords,self.voxel_size)  # coordinates of voxel in World coordinate (N,3)
        cam_pts = rigid_transform(cam_pts, np.linalg.inv(cam_pose))  # world coordiante -> camera coordiante
        pix_z = cam_pts[:, 2]
        pix = self.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1],

        # eliminate pixels outside view frustum
        valid_pix = np.logical_and(pix_x >= 0,
                                   np.logical_and(pix_x < im_w,
                                                  np.logical_and(pix_y > 0,
                                                                 np.logical_and(pix_y < im_h,
                                                                                pix_z > 0))))

        depth_val = np.zeros(len(pix_x))
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix],]
        #print("max of depth val:.{},after valid_pix".format(depth_val.max()))
        # Integrate TSDF
        depth_diff = depth_val - pix_z  # difference
        #print("max od depth diff:.{}".format(depth_diff.max()))
        # TODO: this section, we didn't implement NeuralRecon, we implement KinectFusion
        #valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self.trunc_margin)
        valid_pts = np.logical_and(depth_val > 0,
                                 np.logical_and(self.trunc_margin >= depth_diff, depth_diff >= -self.trunc_margin))
        dist = np.maximum(-1, np.minimum(1, depth_diff / self.trunc_margin))
        #dist = np.minimum(1, depth_diff / self.trunc_margin)
        valid_vox_x = self.vox_coords[
            valid_pts, 0]  # vox_coords have already been considered voxel_size, (N,),(N,),(N,)
        valid_vox_y = self.vox_coords[
            valid_pts, 1]  # vox_coords have already been considered voxel_size, (N,),(N,),(N,)
        valid_vox_z = self.vox_coords[
            valid_pts, 2]  # vox_coords have already been considered voxel_size, (N,),(N,),(N,)
        w_old = self.weight_vol[
            valid_vox_x, valid_vox_y, valid_vox_z]  # self._weight_vol_cpu:np.zeros(nx,ny,nz)  -> np.zeros(N,)
        tsdf_old = self.tsdf_vol[
            valid_vox_x, valid_vox_y, valid_vox_z]  # self._weight_vol_cpu:np.zeros(nx,ny,nz)  -> np.zeros(N,)
        valid_dist = dist[valid_pts]
        tsdf_vol_new, w_new = self.update_tsdf(tsdf_old, valid_dist, w_old, obs_weight)
        self.weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
        self.tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new
        # print("max of tsdf is:{}".format(self.tsdf_vol.max()))
        # print("min of tsdf is:{}".format(self.tsdf_vol.min()))
        self.occ_vol = np.zeros_like(self.tsdf_vol)
        self.occ_vol = np.array(self.occ_vol,dtype = bool)
        self.occ_vol[(self.tsdf_vol < 0.999) & (self.tsdf_vol > -0.999) & (self.weight_vol > 1)] = True
        print("the shape of tsdf and occ is :{},{}".format(self.tsdf_vol.shape,self.occ_vol.shape))
        # integrate color (may not be useful)
        old_color = self.color_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        old_b = np.floor(old_color / self._color_const)
        old_g = np.floor((old_color - old_b * self._color_const) / 256)
        old_r = old_color - old_b * self._color_const - old_g * 256
        new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
        new_b = np.floor(new_color / self._color_const)
        new_g = np.floor((new_color - new_b * self._color_const) / 256)
        new_r = new_color - new_b * self._color_const - new_g * 256
        new_b = np.minimum(255., np.round((w_old * old_b + obs_weight * new_b) / w_new))
        new_g = np.minimum(255., np.round((w_old * old_g + obs_weight * new_g) / w_new))
        new_r = np.minimum(255., np.round((w_old * old_r + obs_weight * new_r) / w_new))
        self.color_vol[valid_vox_x, valid_vox_y, valid_vox_z] = new_b * self._color_const + new_g * 256 + new_r

    def get_volume(self):
        return self.tsdf_vol, self.color_vol, self.weight_vol, self.occ_vol

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume.
        """
        tsdf_vol, color_vol, weight_vol = self.get_volume()

        # Marching cubes
        verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes.
        """
        tsdf_vol, color_vol, weight_vol ,_= self.get_volume()

        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol,level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts * self.voxel_size + self.vol_origin  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
    """
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * np.array([0, max_depth, max_depth, max_depth, max_depth]) /
        cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])  # shape(3,5)
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T  # (5,3) (4,4) --> (5,3) --- >(3,5)
    return view_frust_pts  # (3,5)


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file.
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2],
            norms[i, 0], norms[i, 1], norms[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2],
        ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))
def integrate(
        depth_im,
        cam_intr,
        cam_pose,
        color_im,
        obs_weight,
        world_c,
        vox_coords,
        weight_vol,
        tsdf_vol,
        occ_vol,
        color_vol,
        const,
        sdf_trunc,
        im_h,
        im_w,
):
    # Convert world coordinates to camera coordinates
    if color_im is not None:
        # Fold RGB color image into a single channel image
        color_im = torch.floor(color_im[..., 2] * const + color_im[..., 1] * 256 + color_im[
            ..., 0])  # color_const = 256*256
        #print("the shape of color_im:{}".format(color_im.shape))
        #color_im = color_im.reshape(-1).astype(np.float32)  # faltten  (HxW,)
    else:
        color_im = np.array(0)
    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam.float(), world_c.float().transpose(1, 0)).transpose(1, 0).float()

    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
    pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

    # Eliminate pixels outside view frustum
    valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
    depth_val = torch.zeros(len(pix_x))
    depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

    # Integrate tsdf
    depth_diff = depth_val - pix_z
    dist = torch.clamp(torch.clamp(depth_diff / sdf_trunc, max=1),min=-1)
    valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)&(depth_diff <= sdf_trunc)
    valid_vox_x = vox_coords[valid_pts, 0]
    valid_vox_y = vox_coords[valid_pts, 1]
    valid_vox_z = vox_coords[valid_pts, 2]
    valid_dist = dist[valid_pts]
    w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    w_new = w_old + obs_weight
    tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight * valid_dist) / w_new
    weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
    occ_vol[(tsdf_vol < 0.999) & (tsdf_vol > -0.999) & (weight_vol > 1)] = 1
    #integrate color
    old_color = color_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    old_b = torch.floor(old_color / const)
    old_g = torch.floor((old_color - old_b * const) / 256)
    old_r = old_color - old_b * const - old_g * 256
    new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
    new_b = torch.floor(new_color / const)
    new_g = torch.floor((new_color - new_b * const) / 256)
    new_r = new_color - new_b * const - new_g * 256
    new_b = torch.minimum(torch.tensor(255.), torch.round((w_old * old_b + obs_weight * new_b) / w_new))
    new_g = torch.minimum(torch.tensor(255.), torch.round((w_old * old_g + obs_weight * new_g) / w_new))
    new_r = torch.minimum(torch.tensor(255.), torch.round((w_old * old_r + obs_weight * new_r) / w_new))
    color_vol[valid_vox_x, valid_vox_y, valid_vox_z] = new_b * const + new_g * 256 + new_r
    return weight_vol, tsdf_vol, occ_vol, color_vol


class TSDFVolumeTorch:
    """Volumetric TSDF Fusion of RGB-D Images.
    """

    def __init__(self, vol_bonds, voxel_size, margin=5):
        """Constructor.

        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("[!] No GPU detected. Defaulting to CPU.")
        self.device = torch.device("cpu")

        # Define voxel volume parameters
        self._vol_bonds = torch.tensor(vol_bonds)
        self._voxel_size = torch.max(self._vol_bonds[:,1]-self._vol_bonds[:,0])/64
        self._sdf_trunc = margin * self._voxel_size
        self._const = 256 * 256
        self._integrate_func = integrate

        # Adjust volume bounds
        self._vol_dim = torch.tensor([64,64,64]).long()
        self._vol_bonds[:,1] = self._vol_bonds[:,0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bonds[:,0]

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
        self._tsdf_vol = torch.ones(*self._vol_dim).to(self.device)
        self._weight_vol = torch.zeros(*self._vol_dim).to(self.device)
        self._color_vol = torch.zeros(*self._vol_dim).to(self.device)
        self._occ_vol = torch.zeros(*self._vol_dim).to(self.device)

    def integrate(self,color_im, depth_im, cam_intr, cam_pose, obs_weight):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign to the current observation.
        """
        cam_pose = torch.tensor(cam_pose).float().to(self.device)
        cam_intr = torch.tensor(cam_intr).float().to(self.device)
        depth_im = torch.tensor(depth_im).float().to(self.device)
        color_im = torch.tensor(color_im).float().to(self.device)
        im_h, im_w = depth_im.shape
        weight_vol, tsdf_vol, occ_vol,color_vol= self._integrate_func(
            depth_im,
            cam_intr,
            cam_pose,
            color_im,
            obs_weight,
            self._world_c,
            self._vox_coords,
            self._weight_vol,
            self._tsdf_vol,
            self._occ_vol,
            self._color_vol,
            self._const,
            self._sdf_trunc,
            im_h, im_w,
        )
        self._weight_vol = weight_vol
        self._tsdf_vol = tsdf_vol
        self._occ_vol = occ_vol
        self._color_vol = color_vol
        # Integrate color


    def get_volume(self):
        return self._tsdf_vol, self._weight_vol, self._occ_vol,self._color_vol

    @property
    def sdf_trunc(self):
        return self._sdf_trunc

    @property
    def voxel_size(self):
        return self._voxel_size
        
    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes.
        """
        tsdf_vol, color_vol, _, weight_vol = self.get_volume()
        tsdf_vol = tsdf_vol.cpu().numpy()
        color_vol = color_vol.cpu().numpy()
        weight_vol = weight_vol.cpu().numpy()
        voxel_size = self._voxel_size.cpu().numpy()
        vol_origin = self._vol_origin.cpu().numpy()

        verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts * voxel_size + vol_origin  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._const)
        colors_g = np.floor((rgb_vals - colors_b * self._const) / 256)
        colors_r = rgb_vals - colors_b * self._const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors
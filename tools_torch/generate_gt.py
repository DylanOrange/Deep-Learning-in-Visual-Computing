import os
import sys

import numpy as np

sys.path.append('.')

from fusion import *
import pickle
import argparse
from tqdm import tqdm
from preparation_loader import *
from torch.utils.data import DataLoader

# TODO: right now we don't consider train/test/val split
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
args.save_path = os.path.join(args.data_path, args.save_name)


def save_tsdf(args, scene_path, cam_intr, depth_list, cam_pose_list, color_list, save_mesh=False):
    vol_bnds = np.zeros((3, 2))
    n_imgs = len(depth_list.keys())
    # generate bbox
    if n_imgs > 200:  # original 200
        ind = np.linspace(0, n_imgs - 1, 200).astype(np.int32)
        image_id = np.array(list(depth_list.keys()))[ind]
    else:
        image_id = depth_list.keys()
    for id in image_id:
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        # compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))  # min of xyz
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))  # max of xyz

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume ..")
    tsdf_vol = tsdfvolume(vol_bnds, voxel_size=args.voxel_size, margin=args.margin)

    # Loop all RGB_D images to get TSDF Volume
    for id in depth_list.keys():
        #if id % 100 == 0:
        print("{}: Fusing frame {}/{}".format(scene_path, str(id), str(n_imgs)))
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]
        if len(color_list) == 0:
            color_image = None
        else:
            color_image = color_list[id]

        # integrate the current frame into voxel volume
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1)

    tsdf_info = {
        'vol_origin': tsdf_vol.vol_origin,
        'voxel_size': tsdf_vol.voxel_size,
        'vol_bounds': tsdf_vol.vol_bonds,
    }
    tsdf_path = os.path.join(args.save_path, scene_path)
    if not os.path.exists(tsdf_path):
        os.makedirs(tsdf_path)

    with open(os.path.join(args.save_path, scene_path, 'tsdf_info.pkl'), 'wb') as f:
        pickle.dump(tsdf_info, f)
    tsdf_volume, color_volume, weight_volume, occ_volume = tsdf_vol.get_volume()
    np.savez_compressed(os.path.join(args.save_path, scene_path, 'tsdf'), tsdf_volume)
    np.savez_compressed(os.path.join(args.save_path, scene_path, 'occ'), occ_volume)

    if save_mesh:
        print("Saving mesh to mesh.ply...")
        verts, faces, norms, colors = tsdf_vol.get_mesh()

        meshwrite(os.path.join(args.save_path, scene_path, 'mesh.ply'), verts, faces, norms,
                  colors)


def save_fragment_pkl(args, scene, cam_intr, depth_list,
                      cam_pose_list):  # scene:str  cam_intr:matrix    depth_list:dict   cam_pose_list:
    fragments = []
    print("segment: process scene {}".format(scene))

    # gather pose

    all_ids = []
    ids = []
    all_bnds = []
    count = 0
    last_pose = None
    for id in depth_list.keys():
        depth_im = depth_list[id]
        cam_pose = cam_pose_list[id]

        if count == 0:
            ids.append(id)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.inf
            vol_bnds[:, 1] = -np.inf
            last_pose = cam_pose
            # compute camera view frustum and extend convex hull
            view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:, 1] = np.minimum(vol_bnds[:, 1], np.amin(view_frust_pts, axis=1))
            count += 1
        else:
            angle = np.arccos(
                ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                    [0, 0, 1])).sum())
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = cam_pose
                # compute camera view frustum and extend convex hull
                view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.minimum(vol_bnds[:, 1], np.amin(view_frust_pts, axis=1))
                count += 1
                if count == args.window_size:
                    all_ids.append(ids)
                    all_bnds.append(vol_bnds)
                    ids = []
                    count = 0

        with open(os.path.join(args.save_path, scene, 'tsdf_info.pkl'), 'rb') as f:
            tsdf_info = pickle.load(f)

        # save fragments

        for i, bnds in enumerate(all_bnds):
            #if not os.path.exists(os.path.join(args.save_path, scene, 'fragments', str(i))):
                #os.makedirs(os.path.join(args.save_path, scene, 'fragments', str(i)))
            fragments.append({
                'scene': scene,
                'fragment_id': i,
                'image_ids': all_ids[i],
                'vol_origin': tsdf_info['vol_origin'],
                'voxel_size': tsdf_info['voxel_size'],
                'vol_bonds': bnds
            })
        with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'wb') as f:
            pickle.dump(fragments, f)


def process(args, scannet_files):  # scannet_files contains different scene
    for scene in tqdm(scannet_files):
        if os.path.exists(os.path.join(args.save_path, scene, 'fragments.pkl')):
            continue
        print('read from disk')

        depth_all = {}
        cam_pose_all = {}
        color_all = {}
        n_imgs = len(os.listdir(os.path.join(args.data_path, scene, 'color')))  # number of images
        intrinsic_dir = os.path.join(args.data_path, scene, 'intrinsic', 'intrinsic_depth.txt')
        cam_intr = np.loadtxt(intrinsic_dir, delimiter=' ')[:3, :3]
        dataset = Preparation(n_imgs, scene, args.data_path, args.max_depth)
        dataloader = DataLoader(dataset, batch_size=None, collate_fn=collate_fn,batch_sampler=None)

        for id, (cam_pose, depth_im, color_image) in enumerate(dataloader):
            if id % 100 == 0:
                print("{}:read frame{}/{}".format(scene,str(id),str(n_imgs)))
            if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
                continue
            # print('Dataloader:..')
            # print("max of depth img:.{}".format(depth_im.max()))
            # print("min of depth img:.{}".format(depth_im.min()))
            depth_all.update({id:depth_im})
            cam_pose_all.update({id:cam_pose})
            color_all.update({id:color_image})
        save_tsdf(args,scene,cam_intr,depth_all,cam_pose_all,color_all,save_mesh=True)
        save_fragment_pkl(args,scene,cam_intr,depth_all,cam_pose_all)

if __name__ == '__main__':
    args.data_path = os.path.join(args.data_path,'scans')
    files = sorted(os.listdir(args.data_path))
    process(args,files)


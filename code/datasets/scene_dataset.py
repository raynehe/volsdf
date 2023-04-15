import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
import json

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 ):

        self.instance_dir = os.path.join('/home/rayne/datasets/monosdf', data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)
        # self.all_rays = []

        # if not data_dir == 'fluid':
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        # for i in range(self.n_images):
        #     intrinsic = self.intrinsics_all[i]
        #     pose = self.pose_all[i]
        #     focal = intrinsic[0, 0]
        #     directions = self.get_ray_directions(self.img_res[0], self.img_res[1], focal)
        #     rays_o, rays_d = self.get_rays(directions, pose[:3])
        #     self.all_rays.append(torch.cat([rays_o, rays_d], -1))

        
        # if data_dir == 'fluid':
        #     pose_path = os.path.join(self.instance_dir, 'scene', 'trajectory.npy')
        #     self.pose_all = torch.from_numpy(np.load(pose_path)).float()
        #     intrinsic_path = os.path.join(self.instance_dir, 'intrinsic.npy')
        #     self.intrinsics_all = torch.from_numpy(np.load(intrinsic_path)).float()
        # #     for i in range(self.n_images):
        # #         intrinsic = self.intrinsics_all[i]
        # #         pose = self.pose_all[i]
        # #         focal = intrinsic[0, 0]
        # #         directions = self.get_ray_directions(self.img_res[0], self.img_res[1], focal)
        # #         rays_o, rays_d = self.get_rays(directions, pose[:3])
        # #         self.all_rays.append(torch.cat([rays_o, rays_d], -1))

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            # "rays": self.all_rays[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_ray_directions(self, H, W, focal):
        """
        Get ray directions for all pixels in camera coordinate.
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            H, W, focal: image height, width and focal length

        Outputs:
            directions: (H, W, 3), the direction of the rays in camera coordinate
        """
        from kornia import create_meshgrid
        grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
        i, j = grid.unbind(-1)
        # the direction here is without +0.5 pixel centering as calibration is not so accurate
        # see https://github.com/bmild/nerf/issues/24
        directions = \
            torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

        return directions
    
    def get_rays(self, directions, c2w):
        """
        Get ray origin and normalized directions in world coordinate for all pixels in one image.
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            directions: (H, W, 3) precomputed ray directions in camera coordinate
            c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

        Outputs:
            rays_o: (H*W, 3), the origin of the rays in world coordinate
            rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
        """
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ c2w[:, :3].T # (H, W, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        # The origin of all rays is the camera origin in world coordinate
        rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

        # rays_d = rays_d.view(-1, 3)
        # rays_o = rays_o.view(-1, 3)

        return rays_o, rays_d
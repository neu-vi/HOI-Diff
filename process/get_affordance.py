import numpy as np
import os
import sys
sys.path.append('./')
import argparse
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from scipy.spatial import cKDTree
from visualize.plot_script import plot_3d_motion
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='behave', choices=['behave', 'omomo'])
args = parser.parse_args()

if args.dataset == 'behave':
    data_path = './dataset/behave_t2m'
    motion_path = './dataset/behave_t2m/sequences'
    object_path = './dataset/behave_t2m/object_mesh'
else:
    data_path = './dataset/omomo_t2m'
    motion_path = './dataset/omomo_t2m/sequences'
    object_path = './dataset/omomo_t2m/object_mesh'

import random

# motion_list = random.sample(os.listdir(motion_path), 10)
motion_list = os.listdir(motion_path)
for seq_name in tqdm(motion_list):
    human_jts = np.load(os.path.join(motion_path, seq_name, 'human_motion.npz'))['jts']

    with np.load(os.path.join(motion_path, seq_name, 'object_motion.npz')) as data:
        obj_angles = data['angles']
        obj_trans = data['trans']
        obj_name = str(data['name'])

    # obj_sample_points = np.load(os.path.join(object_path, obj_name, 'sample_points.npy'))
    obj_sample_idx = np.load(os.path.join(data_path, 'sample_objids', obj_name, f'{obj_name}.npy'))




    mesh_obj = trimesh.load(os.path.join(object_path, obj_name, f"{obj_name}.obj"))
    obj_sample_points = np.array(mesh_obj.vertices)[obj_sample_idx]


    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts_motion = np.matmul(obj_sample_points[None, ...], np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]
   # Find the nearest mesh vertex for each sampled point
    all_obj_contact_indices = []
    all_contact_points = []
    all_contact_mask = []




    for t_idx in range(human_jts.shape[0]):
        tree = cKDTree(obj_verts_motion[t_idx])
        dist, _ = tree.query(human_jts[t_idx], k=1)
        contact_mask = dist < 0.17
        all_contact_mask.append(contact_mask)
        
    all_contact_mask = np.stack(all_contact_mask, axis=0)


    human_contact_count = np.zeros(human_jts.shape[1])
    for i in range(human_jts.shape[0]):
        for j in range(human_jts.shape[1]):
            if all_contact_mask[i, j]:
                human_contact_count[j] += 1

    h_density = human_contact_count / human_contact_count.max()
    h_stable_contact_mask = h_density > 0.5  # 取前30%密度点

    human_contact_idx = np.where(h_stable_contact_mask)[0]


    # print(f"human_contact_idx: {human_contact_idx.shape}")
    # for t_idx in range(human_jts.shape[0]):
    #     tree = cKDTree(obj_verts_motion[t_idx])
    #     dist, obj_contact_indices = tree.query(human_jts[t_idx, human_contact_idx], k=1)

    #     all_obj_contact_indices.append(obj_contact_indices)
    #     all_contact_points.append(obj_verts_motion[t_idx, obj_contact_indices, :])
        
    # all_obj_contact_indices = np.stack(all_obj_contact_indices, axis=0)
    # all_contact_points = np.stack(all_contact_points, axis=0)

    all_obj_contact_idx= []
    for contact_idx in human_contact_idx:
        obj_count = np.zeros(len(obj_sample_points))
        for t_idx in range(human_jts.shape[0]):
            dist = torch.cdist(torch.from_numpy(obj_verts_motion[t_idx]).unsqueeze(0).float(), torch.from_numpy(human_jts[t_idx, contact_idx]).unsqueeze(0).float())
            min_dist_idx = torch.argmin(dist.squeeze().squeeze(), dim=-1)
            obj_count[min_dist_idx] += 1

        o_density = obj_count / obj_count.max()
        o_stable_idx = np.argmax(o_density)
        all_obj_contact_idx.append(o_stable_idx)



    affordance_data = np.zeros((human_jts.shape[1], 4))
    affordance_data[human_contact_idx, 0] = 1.0
    affordance_data[human_contact_idx, 1:] = obj_sample_points[all_obj_contact_idx]


    os.makedirs(os.path.join(data_path, 'affordance'), exist_ok=True)
    np.save(os.path.join(data_path, 'affordance', f'{seq_name}.npy'), affordance_data)


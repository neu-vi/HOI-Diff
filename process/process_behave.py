'''
Some codes are adapted from: 
    Author: Sirui Xu et al.
    Cite: InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation
'''


import json
import os
import sys
sys.path.append('./')
import os.path
import numpy as np
import torch
from tqdm import tqdm
import smplx
import trimesh
from scipy.spatial.transform import Rotation

from bps_torch.bps import bps_torch

anno_path = './utils/action_label.json'

with open(anno_path, 'r') as f:
    anno = json.load(f)

id2label = {}
for des in anno['label_description']:
    id2label[des['id']] = des['description']

all_sequence = []
sequence = {}

action_label = anno['action_label']
for action in action_label:
    action_id = action['label']
    action_label = id2label[action_id]
    if action_label == 'no_interaction' or action_label == 'action_transition':
        if len(sequence) > 0:
            all_sequence.append(sequence)
            sequence = {}
        continue

    if (len(sequence) > 0) and (action_label != sequence['action']):
        all_sequence.append(sequence)
        sequence = {}

    seq_name = action['name']
    frame = int(action['frame'][1:].split('.')[0])

    if len(sequence) == 0:
        sequence = {'name': seq_name,
                    'action': action_label,
                    'frame' :[frame]}
    else:
        sequence['frame'].append(frame)




smpl_model_male = smplx.create('./body_models', model_type='smplh',
                          gender="male",
                          use_pca=False,
                          ext='pkl')

smpl_model_female = smplx.create('./body_models', model_type='smplh',
                          gender="female",
                          use_pca=False,
                          ext='pkl')

smpl = {'male': smpl_model_male, 'female': smpl_model_female}



motion_path = './dataset/raw_data/behave_raw'
object_path = './dataset/raw_data/behave_objects'

new_data_path = './dataset/behave_t2m'
new_object_path = os.path.join(new_data_path, 'object_mesh')
os.makedirs(new_object_path, exist_ok=True)
new_motion_path = os.path.join(new_data_path, 'sequences')
os.makedirs(new_motion_path, exist_ok=True)



print(f"processing object meshes...")
for obj_name in os.listdir(object_path):
    print(f"processing {obj_name}")
    obj_save_path = os.path.join(new_object_path, obj_name)
    os.makedirs(obj_save_path, exist_ok=True)
    mesh_obj = trimesh.load(os.path.join(object_path, f"{obj_name}/{obj_name}.obj"), force='mesh')
    obj_verts = mesh_obj.vertices
    obj_faces = mesh_obj.faces
    mesh_obj.export(os.path.join(obj_save_path, f"{obj_name}.obj"))



    bps_th = bps_torch()
    bps_obj = np.load(os.path.join('./dataset', 'bps_basis_set_1024_1.npy'))
    bps_obj = torch.from_numpy(bps_obj).float().cuda()


    bps_obj_verts = (obj_verts)[None, ...]
    torch_obj_verts = torch.from_numpy(bps_obj_verts).float().cuda()
    
    bps_object_geo = bps_th.encode(x=torch_obj_verts, \
            feature_type=['deltas'], \
            custom_basis=bps_obj[None,...])['deltas'] # T X N X 3 
    bps_object_geo_np = bps_object_geo.data.detach().cpu().numpy()
    
    np.save(os.path.join(obj_save_path, "bps_1024.npy"), bps_object_geo_np)



    points, face_indices = trimesh.sample.sample_surface(
        mesh_obj,
        count=1024       # let trimesh choose automatically
        )

    # Find the nearest mesh vertex for each sampled point
    closest_vertex_indices = mesh_obj.kdtree.query(points)[1]


    closest_vertex_indices = np.load(os.path.join(new_data_path, f"sample_objids/{obj_name}/{obj_name}.npy"))

    np.save(os.path.join(obj_save_path, "sample_points.npy"), obj_verts[closest_vertex_indices])

    os.makedirs(os.path.join(new_data_path, f"sample_objids/{obj_name}"), exist_ok=True)
    np.save(os.path.join(new_data_path, f"sample_objids/{obj_name}/{obj_name}.npy"), closest_vertex_indices)



pbar = tqdm(all_sequence)
pbar.set_description('Processing BEHAVE motion data')
for seq in pbar:

    # print(f"processing {seq['name']}")
    name = seq['name']
    seq_name_fine = seq['name'] + '_{}'.format(seq['frame'][0])
    min_frame = seq['frame'][0]
    max_frame = seq['frame'][-1]

    if not os.path.exists(os.path.join(motion_path, name, 'object_fit_all.npz')):
        continue
    if not os.path.exists(os.path.join(motion_path, name, 'smpl_fit_all.npz')):
        continue

    if os.path.exists(os.path.join(new_motion_path, seq_name_fine, 'human_motion.npz')): # already processed
        continue

    with np.load(os.path.join(motion_path, name, 'object_fit_all.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, frame_times = f['angles'], f['trans'], f['frame_times']
    with np.load(os.path.join(motion_path, name, 'smpl_fit_all.npz'), allow_pickle=True) as f:
        poses, betas, trans = f['poses'], f['betas'], f['trans']


    pose_h_ = []
    betas_h_ = []
    trans_h_ = []
    frame_times_h_ = []
    angles_o_ = []
    trans_o_ = []
    frame_times_o_ = []
    for ind, ft in enumerate(frame_times):
        ft = float(ft[1:])
        if ft > (min_frame - 0.5) and ft < (max_frame + 0.5):
            # print(f"frame {ind} {ft}")
            pose_h_.append(poses[ind])
            betas_h_.append(betas[ind])
            trans_h_.append(trans[ind])

            angles_o_.append(obj_angles[ind])
            trans_o_.append(obj_trans[ind])


    if len(pose_h_) == 0:
        continue  # skip the sequence if no human motion data is available

    poses=np.stack(pose_h_, axis=0)
    betas=np.stack(betas_h_, axis=0)
    trans=np.stack(trans_h_, axis=0)
    obj_angles=np.stack(angles_o_, axis=0)
    obj_trans=np.stack(trans_o_, axis=0)
    
    # frame_times = frame_times.shape[0]
    info_file = os.path.join(motion_path, name, 'info.json')
    info = json.load(open(info_file))
    gender = info['gender']
    obj_name = info['cat']
    
    smpl_model = smpl[gender]
    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                              global_orient=torch.from_numpy(poses[:, :3]).float(),
                              left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                              right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                              betas=torch.from_numpy(betas).float(),
                              transl=torch.from_numpy(trans).float(),)
    pelvis = smplx_output.joints.detach().numpy()[:, 0, :]
    rotvecs = poses[:, :3]
    rotations = Rotation.from_rotvec(rotvecs)
    rotation_matrix_x = Rotation.from_euler('x', -np.pi, degrees=False)
    # Apply the rotation to the batch of rotations
    rotated_rotations = rotation_matrix_x * rotations
    # Convert the rotated rotations back to rotation vectors
    poses[:, :3] = rotated_rotations.as_rotvec()

    trans = rotation_matrix_x.apply(trans)

    rotvecs2 = obj_angles
    rotations2 = Rotation.from_rotvec(rotvecs2)

    # Apply the rotation to the batch of rotations
    rotated_rotations2 = rotation_matrix_x * rotations2
    # Convert the rotated rotations back to rotation vectors
    obj_angles = rotated_rotations2.as_rotvec()
    obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis)

    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                              global_orient=torch.from_numpy(poses[:, :3]).float(),
                              left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                              right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                              betas=torch.from_numpy(betas).float(),
                              transl=torch.from_numpy(trans).float(),)
    
    verts = smplx_output.vertices.detach().numpy()
    pelvis = smplx_output.joints.detach().numpy()[:, 0, :]

    jts = smplx_output.joints.detach().numpy()[:, :22, :]
    faces = smpl_model.faces
    
    obj_trans = pelvis + obj_trans_delta

    mesh_obj = trimesh.load(os.path.join(object_path, f"{obj_name}/{obj_name}.obj"), force='mesh')
    obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces
    mesh_obj.vertices = (obj_verts - obj_verts.mean(axis=0, keepdims=True))

    # mesh_obj.export(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"))
    # obj_verts = mesh_obj.vertices[None, ...]
    obj_verts = np.load(os.path.join(new_object_path, f"{obj_name}/sample_points.npy"))
    obj_verts = obj_verts[None, ...]


    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    diff_fix = min(verts[:30, ..., 1].min(), obj_verts[:30, ..., 1].min())
    obj_trans[..., 1] -= diff_fix
    trans[..., 1] -= diff_fix
    jts[..., 1] -= diff_fix

    obj = {
        'angles': obj_angles,
        'trans': obj_trans,
        'name': obj_name,
    }
    human = {
        'poses': poses,
        'betas': betas[0],
        'trans': trans,
        'gender': gender,
        'jts': jts,
    }


    os.makedirs(os.path.join(new_motion_path, seq_name_fine), exist_ok=True)
    np.savez(os.path.join(new_motion_path, seq_name_fine, 'object_motion.npz'), **obj)
    np.savez(os.path.join(new_motion_path, seq_name_fine, 'human_motion.npz'), **human)

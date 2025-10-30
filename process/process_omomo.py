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
import joblib
from scipy.spatial.transform import Rotation
from os.path import join as pjoin
from bps_torch.bps import bps_torch
from visualize.plot_script import plot_3d_motion


smplx_model_male = smplx.create('./body_models', model_type='smplx',
                          gender="male",
                          use_pca=False,
                          num_betas=16,
                          ext='pkl')

smplx_model_female = smplx.create('./body_models', model_type='smplx',
                          gender="female",
                          use_pca=False,
                          num_betas=16,
                          ext='pkl')
smpl_model_neutral = smplx.create('./body_models', model_type='smplx',
                          gender="neutral",
                          use_pca=False,
                          num_betas=16,
                          ext='pkl')

smplx_model = {'male': smplx_model_male, 'female': smplx_model_female, 'neutral': smpl_model_neutral}



motion_path = './dataset/raw_data/omomo_raw'
object_path = './dataset/raw_data/omomo_objects'

new_data_path = './dataset/omomo_t2m'
new_object_path = os.path.join(new_data_path, 'object_mesh')
os.makedirs(new_object_path, exist_ok=True)
new_motion_path = os.path.join(new_data_path, 'sequences')
os.makedirs(new_motion_path, exist_ok=True)


obj_scale_list = {'clothesstand': 0.27487725, 'floorlamp': 0.3792082, 'largebox': 0.3503452, 
'smallbox': 0.3137573, 'tripod': 0.22367683, 'vacuum': 0.19233231, 'whitechair': 0.3136575, 
'monitor': 0.15750739, 'mop': 0.16737443, 'trashcan': 0.233327, 'plasticbox': 0.02526473, 
'smalltable': 0.016271079, 'woodchair': 0.025014637, 'suitcase': 0.36519036, 'largetable': 0.025855122}


print(f"processing object meshes...")
for obj_name in os.listdir(object_path):
    print(f"processing {obj_name}")
    new_obj_name = obj_name.split('_')[0]
    obj_save_path = os.path.join(new_object_path, new_obj_name)
    os.makedirs(obj_save_path, exist_ok=True)
    mesh_obj = trimesh.load(os.path.join(object_path, f"{new_obj_name}_cleaned_simplified.obj"), force='mesh')
    obj_verts = np.array(mesh_obj.vertices) * obj_scale_list[new_obj_name]
    obj_faces = np.array(mesh_obj.faces)
    new_mesh_obj = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
    new_mesh_obj.export(os.path.join(obj_save_path, f"{new_obj_name}.obj"))



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



    # points, face_indices = trimesh.sample.sample_surface(
    #     mesh_obj,
    #     count=1024       # let trimesh choose automatically
    #     )

    # # Find the nearest mesh vertex for each sampled point
    # closest_vertex_indices = mesh_obj.kdtree.query(points)[1]

    closest_vertex_indices = np.load(os.path.join(new_data_path, f"sample_objids/{new_obj_name}/{new_obj_name}.npy"))

    np.save(os.path.join(obj_save_path, "sample_points.npy"), obj_verts[closest_vertex_indices])

    os.makedirs(os.path.join(new_data_path, f"sample_objids/{new_obj_name}"), exist_ok=True)
    np.save(os.path.join(new_data_path, f"sample_objids/{new_obj_name}/{new_obj_name}.npy"), closest_vertex_indices)



seq_data_path = pjoin(motion_path, "train_diffusion_manip_seq_joints24.p")
seq_data_path2 = pjoin(motion_path, "test_diffusion_manip_seq_joints24.p")
data_dict1 = joblib.load(seq_data_path)
data_dict2 = joblib.load(seq_data_path2)

# data_dict = [data_dict1, data_dict2]
idx_list1 = np.arange(len(data_dict1))
idx_list2 = np.arange(len(data_dict1), len(data_dict1)+len(data_dict2))


dataset = 'omomo'
obj_scale_list = {}
id_list = [idx_list1, idx_list2]
for i, dic in list(enumerate([data_dict1, data_dict2])):
    sub_ids = id_list[i]
    for index in tqdm(dic):
        idx = '{:04d}'.format(sub_ids[index])
        seq_name = dic[index]['seq_name']
        obj_name = dic[index]['seq_name'].split("_")[1]

        # if os.path.exists(os.path.join(new_motion_path, seq_name, 'human_motion.npz')): # already processed
        #     continue
        


        betas = dic[index]['betas'] # 16 
        gender = str(dic[index]['gender'])
        pose_trans = dic[index]['trans'] # T X 3 
        pose_root = dic[index]['root_orient'] # T X 3 
        pose_body = dic[index]['pose_body'] # T X 63
        pose_hands = np.zeros([pose_trans.shape[0], 90]) # T X 90
        pose_body = np.concatenate([pose_body, pose_hands], -1) # T X 153
        frame_num = pose_trans.shape[0]
        obj_trans = dic[index]['obj_trans'][:, :, 0] # T X 3
        obj_rot = dic[index]['obj_rot'] # T X 3 X 3 

        obj_scale = dic[index]['obj_scale'].mean() # T
        
        obj_scale_list[str(obj_name)] = obj_scale

        obj_mesh_path = pjoin(object_path, obj_name+"_cleaned_simplified.obj")


        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_verts = np.asarray(mesh.vertices)  # Nv X 3
        obj_verts *= obj_scale
        obj_faces = np.asarray(mesh.faces) # Nf X 3 
            
        obj_angles = Rotation.from_matrix(obj_rot).as_rotvec()


        poses = np.concatenate([pose_root, pose_body], -1)


        body_model = smplx_model[gender]
        smplx_output = body_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                global_orient=torch.from_numpy(poses[:, :3]).float(),
                                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                jaw_pose=torch.zeros([poses.shape[0], 3]).float(),
                                leye_pose=torch.zeros([poses.shape[0], 3]).float(),
                                reye_pose=torch.zeros([poses.shape[0], 3]).float(),
                                expression=torch.zeros(poses.shape[0], 10).float(),
                                betas=torch.from_numpy(betas).float(),
                                transl=torch.from_numpy(pose_trans).float(),)


        pelvis = smplx_output.joints.detach().numpy()[:, 0, :]
        rotvecs = poses[:, :3]
        rotations = Rotation.from_rotvec(rotvecs)
        rotation_matrix_x = Rotation.from_euler('x', -np.pi/2, degrees=False)
        # Apply the rotation to the batch of rotations
        rotated_rotations = rotation_matrix_x * rotations
        # Convert the rotated rotations back to rotation vectors
        poses[:, :3] = rotated_rotations.as_rotvec()

        pose_trans = rotation_matrix_x.apply(pose_trans)

        rotvecs2 = obj_angles
        rotations2 = Rotation.from_rotvec(rotvecs2)

        # Apply the rotation to the batch of rotations
        rotated_rotations2 = rotation_matrix_x * rotations2
        # Convert the rotated rotations back to rotation vectors
        obj_angles = rotated_rotations2.as_rotvec()
        obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis)



        smplx_output = body_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                global_orient=torch.from_numpy(poses[:, :3]).float(),
                                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                jaw_pose=torch.zeros([poses.shape[0], 3]).float(),
                                leye_pose=torch.zeros([poses.shape[0], 3]).float(),
                                reye_pose=torch.zeros([poses.shape[0], 3]).float(),
                                expression=torch.zeros(poses.shape[0], 10).float(),
                                betas=torch.from_numpy(betas).float(),
                                transl=torch.from_numpy(pose_trans).float(),)


        verts = smplx_output.vertices.detach().numpy()
        pelvis = smplx_output.joints.detach().numpy()[:, 0, :]

        jts = smplx_output.joints.detach().numpy()[:, :22, :]
        faces = body_model.faces
        
        obj_trans = pelvis + obj_trans_delta



        sample_idx = np.load(os.path.join(new_data_path, f"sample_objids/{obj_name}/{obj_name}.npy"))
        sample_verts = obj_verts[sample_idx]

        rot_mat = Rotation.from_rotvec(obj_angles).as_matrix()
        obj_verts = np.matmul(obj_verts[np.newaxis], rot_mat.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + obj_trans[:, np.newaxis]

        sample_verts = np.matmul(sample_verts[np.newaxis], rot_mat.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + obj_trans[:, np.newaxis]



        # plot_3d_motion(os.path.join('./vis_check/', dataset, seq_name+ '.mp4'), None, jts, [sample_verts],  title='test', figsize=(10, 10), fps=20, radius=4)



        diff_fix = min(verts[:30, ..., 1].min(), obj_verts[:30, ..., 1].min())
        obj_trans[..., 1] -= diff_fix
        pose_trans[..., 1] -= diff_fix
        jts[..., 1] -= diff_fix

        obj = {
            'angles': obj_angles,
            'trans': obj_trans,
            'name': obj_name,
        }
        human = {
            'poses': poses,
            'betas': betas[0],
            'trans': pose_trans,
            'gender': gender,
            'jts': jts,
        }



        os.makedirs(os.path.join(new_motion_path, seq_name), exist_ok=True)
        np.savez(os.path.join(new_motion_path, seq_name, 'object_motion.npz'), **obj)
        np.savez(os.path.join(new_motion_path, seq_name, 'human_motion.npz'), **human)






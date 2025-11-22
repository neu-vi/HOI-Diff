import numpy as np
import os
import sys
import smplx
sys.path.append('./')
import argparse
import trimesh
import random
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from scipy.spatial import cKDTree
from visualize.plot_script import plot_3d_motion
import torch
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='behave', choices=['behave', 'omomo'])
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



markerset_smplx = [
    # body (38)
    4391, 5533, 5761, 4509, 5678, 4245, 4379, 4515, 5726, 8852,
    4258, 3638, 3781, 3705, 3479, 4039, 4297, 5615, 8455, 7179,
    7145, 7028, 7115, 7251, 8421, 8634, 7036, 6401, 6539, 6466,
    6352, 6778, 5532, 7293, 7274, 4557, 4538, 5944,

    # foot (6)
    5893, 5899, 5857, 8587, 8593, 8551,

    # head (5)
    3035, 2148, 2041, 3076, 9002,


    # butt
    3462, 5574, 6223,

    # finger (30)
    4875, 4897, 4931, 5014, 5020, 5045,
    5242, 5250, 5268, 5124, 5131, 5149,
    4683, 5321, 5346, 7611, 7633, 7667,
    7750, 7756, 7781, 7978, 7984, 8001,
    7860, 7867, 7884, 7419, 7602, 8082,

    # hand (6)
    4686, 7423, 4748, 4615, 7500, 7351,



    # palm (44)
    4628, 4641, 4660, 4690, 4691, 4710,
    4750, 4885, 4957, 4970, 5001, 5012,
    5082, 5111, 5179, 5193, 5229, 5296,
    5306, 5315, 5353, 5387, 7357, 7396,
    7443, 7446, 7536, 7589, 7618, 7625,
    7692, 7706, 7730, 7748, 7789, 7847,
    7858, 7924, 7931, 7976, 8039, 8050,
    8087, 8122,
]



markerset_smplh= [
    751,  3506, 3453, 761, 3095, 1724, 1727, 1422, 1322, 3458,
    1661, 1083, 1087, 3189, 979, 791, 3481, 3149, 6853, 6366, 
    6505, 6471, 5196, 4896, 6877, 6860, 5131, 4568, 4641, 6589,
    4519, 5328, 1326, 5419, 5447, 1958, 1988, 6472, 
    
    # foot
    3355, 3363, 3358, 6755, 6763, 6758, 
    # head
    3660, 149, 2972, 4300, 445, 
    
    # butt
    1462, 4384, 4931, 
    
    # finger
    2556, 2445, 2445, 2556, 2556, 2445, 2680, 2680, 2680, 2556, 
    2556, 2556, 1987, 2735, 2718, 6016, 5905, 5905, 6016, 6016,
    6016, 6140, 6140, 6140, 6133, 6016, 6016, 6194, 6179, 6192, 
    
    # hand
    2096, 5556, 2095, 2077, 5446, 5538,
    # palm
    2693, 2556, 2556, 2693, 2680, 2555, 2680, 2556, 2445, 2445, 
    2556, 2556, 2556, 2556, 2556, 2556, 2680, 2680, 2680, 2556,
    2556, 2680, 
    6154, 6017, 6154, 6195, 6016, 6154, 6133, 6140, 6016, 6016, 
    6016, 6016, 6016, 6016, 6016, 6016, 6140, 6140, 6140, 6196, 
    6192, 6140
    ]


markerset_smplx_dict = {

    "back": [
        4391, 4509, 7179,
    ],

    "left_shoulder": [4039],
    "right_shoulder": [7251],

    "left_foot": [
        5893, 5899, 5857
    ],
    "right_foot": [
       8587, 8593, 8551,
    ],

    "butt": [
        3462, 5574, 6223,
    ],
    "left_hand": [
       4628, 4641, 4660, 4690, 4691, 4710,
        4750, 4885, 4957, 4970, 5001, 5012,
        5082, 5111, 5179, 5193, 5229, 5296,
        5306, 5315, 5353, 5387,
    ],
    "right_hand": [
      7357, 7396,7443, 7446, 7536, 7589, 7618, 7625,
        7692, 7706, 7730, 7748, 7789, 7847,
        7858, 7924, 7931, 7976, 8039, 8050,
        8087, 8122
    ],
}


markerset_smplh_dict = {
    # "body": [ 
    #      5196, 4896, 6877, 6860, 5131, 4568, 4641, 6589,
    #     4519, 5328,  5419, 5447,  6472,
    # ],

    "back": [
        751, 761, 6366,
    ],

    "left_shoulder": [791],
    "right_shoulder": [4896],

    "left_foot": [
        3355, 3363, 3358,
    ],
    "right_foot": [
        6755, 6763, 6758,
    ],

    # "head": [
    #     3660, 149, 2972, 4300, 445,
    # ],

    "butt": [
        1462, 4384, 4931,
    ],
    "left_hand": [
        2013, 2122, 2095, 2096, 2080, 2096, 
        2077, 2773, 2344, 2344, 2777, 2567, 
        2574, 2771, 2691, 2691, 2773, 2688, 
        2690, 1985, 2742, 2077, 
    ],
    "right_hand": [
       5556, 5557, 5556, 5446, 6203, 5581, 
        5540, 5538, 5694, 5805, 6232, 6145, 
        6028, 6093, 6149, 6151, 6093, 6149, 
        6149, 5446, 6204, 5538
    ],
}





if args.dataset == 'behave':
    data_path = './dataset/behave_t2m'
    motion_path = './dataset/behave_t2m/sequences'
    object_path = './dataset/behave_t2m/object_mesh'
    text_path = './dataset/behave_t2m/texts'
else:
    data_path = './dataset/omomo_t2m'
    motion_path = './dataset/omomo_t2m/sequences'
    object_path = './dataset/omomo_t2m/object_mesh'
    text_path = './dataset/omomo_t2m/texts'



smplh_model_male = smplx.create('./body_models', model_type='smplh',
                        gender="male",
                        use_pca=False,
                        flat_hand_mean=True,
                        ext='pkl').to(device)

smplh_model_female = smplx.create('./body_models', model_type='smplh',
                        gender="female",
                        use_pca=False,
                        flat_hand_mean=True,
                        ext='pkl').to(device)

smplh_models = {'male': smplh_model_male, 'female': smplh_model_female}


smplx_model_male = smplx.create('./body_models', model_type='smplx',
                          gender="male",
                          use_pca=False,
                          num_betas=16,
                          ext='pkl').to(device)

smplx_model_female = smplx.create('./body_models', model_type='smplx',
                          gender="female",
                          use_pca=False,
                          num_betas=16,
                          ext='pkl').to(device)

smpl_model_neutral = smplx.create('./body_models', model_type='smplx',
                          gender="neutral",
                          use_pca=False,
                          num_betas=16,
                          ext='pkl').to(device)

smplx_models = {'male': smplx_model_male, 'female': smplx_model_female, 'neutral': smpl_model_neutral}
# motion_list = random.sample(os.listdir(motion_path), 10)
motion_list = os.listdir(motion_path)
for seq_name in tqdm(motion_list):
    human_jts = np.load(os.path.join(motion_path, seq_name, 'human_motion.npz'))['jts']


    with np.load(os.path.join(motion_path, seq_name, 'human_motion.npz')) as f:
        poses, betas, trans, betas, gender = f['poses'], f['betas'], f['trans'], f['betas'], str(f['gender'])


    markerset = markerset_smplh if args.dataset == 'behave' else markerset_smplx
    markerset_dict = markerset_smplh_dict if args.dataset == 'behave' else markerset_smplx_dict
    smpl_model = smplh_models[gender] if args.dataset == 'behave' else smplx_models[gender]
    if args.dataset == 'behave':
        smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float().to(device),
                                global_orient=torch.from_numpy(poses[:, :3]).float().to(device),
                                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float().to(device),
                                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float().to(device),
                                betas=torch.from_numpy(betas).unsqueeze(0).repeat(poses.shape[0], 1).float().to(device),
                                transl=torch.from_numpy(trans).float().to(device),)
    else:
        smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float().to(device),
                                global_orient=torch.from_numpy(poses[:, :3]).float().to(device),
                                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float().to(device),
                                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float().to(device),
                                jaw_pose=torch.zeros([poses.shape[0], 3]).float().to(device),
                                leye_pose=torch.zeros([poses.shape[0], 3]).float().to(device),
                                reye_pose=torch.zeros([poses.shape[0], 3]).float().to(device),
                                expression=torch.zeros(poses.shape[0], 10).float().to(device),
                                betas=torch.from_numpy(betas).unsqueeze(0).repeat(poses.shape[0], 1).float().to(device),
                                transl=torch.from_numpy(trans).float().to(device))


    human_verts = smplx_output.vertices.detach().cpu().numpy()
    markers = human_verts[:, markerset,:]




    with np.load(os.path.join(motion_path, seq_name, 'object_motion.npz')) as data:
        obj_angles = data['angles']
        obj_trans = data['trans']
        obj_name = str(data['name'])

    obj_sample_idx = np.load(os.path.join(data_path, 'sample_objids', obj_name, f'{obj_name}.npy'))




    mesh_obj = trimesh.load(os.path.join(object_path, obj_name, f"{obj_name}.obj"))
    obj_sample_points = np.array(mesh_obj.vertices)[obj_sample_idx]


    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts_motion = np.matmul(obj_sample_points[None, ...], np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]
   # Find the nearest mesh vertex for each sampled point
    all_obj_contact_indices = []
    all_contact_points = []
    all_contact_mask = []

    # check visualization
    # plot_3d_motion('./{}.mp4'.format(seq_name), None, markers, [obj_verts_motion], title=str(seq_name))




    for t_idx in range(markers.shape[0]):
        tree = cKDTree(obj_verts_motion[t_idx])
        dist, _ = tree.query(markers[t_idx], k=1)
        contact_mask = dist < 0.1
        all_contact_mask.append(contact_mask)
        
    all_contact_mask = np.stack(all_contact_mask, axis=0)



    human_contact_count = np.zeros(markers.shape[1])
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            if all_contact_mask[i, j]:
                human_contact_count[j] += 1

    h_density = human_contact_count / human_contact_count.max()
    h_stable_contact_mask = h_density > 0.3  # get the top 50 % 

    human_contact_idx = np.where(h_stable_contact_mask)[0]



    all_obj_contact_idx= []
    for contact_idx in human_contact_idx:
        obj_count = np.zeros(len(obj_sample_points))
        for t_idx in range(markers.shape[0]):
            dist = torch.cdist(torch.from_numpy(obj_verts_motion[t_idx]).unsqueeze(0).float(), torch.from_numpy(markers[t_idx, contact_idx]).unsqueeze(0).float())
            min_dist_idx = torch.argmin(dist.squeeze().squeeze(), dim=-1)
            obj_count[min_dist_idx] += 1

        o_density = obj_count / obj_count.max()
        o_stable_idx = np.argmax(o_density)
        all_obj_contact_idx.append(o_stable_idx)



    affordance_data = np.zeros((8, 4))


    # As stated in the paper, we only consider 8 primary body joints for contact
    # 0 -pelvis, 9-top back, 10,11-feet, 16,17-shoulders, 20,21-wrists
    key_body_index = {"butt": 0, "back": 1, 
            "left_foot": 2, "right_foot": 3, 
            "left_shoulder": 4, "right_shoulder": 5,
             "left_hand": 6, "right_hand": 7}  # pelvis, top back, feet, shoulders, wrists

    # mapping contact markers to key body joints
    part_state = {part: False for part in key_body_index.keys()}
    for i, h_dx in enumerate(np.array(markerset)[np.array(human_contact_idx)]):

        for part in markerset_dict:
            if h_dx in markerset_dict[part] and not part_state[part]:
                obj_idx = all_obj_contact_idx[i]
  
                joint_idx = key_body_index[part]
                affordance_data[joint_idx, 0] = 1.0
                affordance_data[joint_idx, 1:] = obj_sample_points[obj_idx]
                part_state[part] = True
                break

    os.makedirs(os.path.join(data_path, 'affordance'), exist_ok=True)
    np.save(os.path.join(data_path, 'affordance', f'{seq_name}.npy'), affordance_data)




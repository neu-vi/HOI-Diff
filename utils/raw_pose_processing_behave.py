# adapted from HumanML3D, Guo et al. 2021



import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation


from human_body_prior.tools.omni_tools import copy2cpu as c2c

os.environ['PYOPENGL_PLATFORM'] = 'egl'



# %%
# Choose the device to run the body model on.
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
from human_body_prior.body_model.body_model import BodyModel

male_bm_path = './body_models/smplh/SMPLH_MALE.npz'

female_bm_path = './body_models/smplh/SMPLH_FEMALE.npz'


num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas).to(comp_device)
faces = c2c(male_bm.f)

female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas).to(comp_device)

paths = []
folders = []
dataset_names = []
for root, dirs, files in os.walk('./dataset/raw_behave'):
    folders.append(root)
    for name in files:
        dataset_name = root.split('/')[2]
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        paths.append(os.path.join(root, files[-1]))




# %%
save_root = './dataset/joints_behave'
save_folders = [folder.replace('./dataset/raw_behave', './dataset/joints_behave') for folder in folders]
for folder in save_folders:
    os.makedirs(folder, exist_ok=True)
group_path = [[path for path in paths if name in path] for name in dataset_names]

# %%
trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0]])
ex_fps = 30
def behave_to_pose(src_path, save_path):
    seq_info_path = src_path.replace("smpl_fit_all.npz", "info.json")
    if os.path.exists(seq_info_path):
        with open(seq_info_path, "r") as f:
            seq_info = json.load(f)
        gender = seq_info["gender"]
    else:
        gender = 'neutral'

    src_path_obj = src_path.replace('smpl_fit_all.npz', 'object_fit_all.npz')
    fps = 30
    bdata = np.load(src_path, allow_pickle=True)
    print(f"bdata: {bdata['poses'].shape}")
    bdata_obj = np.load(src_path_obj, allow_pickle=True)

    frame_number = bdata['trans'].shape[0]

    fId = 0 # frame id of the mocap sequence
    pose_seq = []
    if gender == 'male':
        bm = male_bm
    else:
        bm = female_bm
    down_sample = int(fps / ex_fps)

    
    with torch.no_grad():
        for fId in range(0, frame_number, down_sample):
            root_orient = torch.Tensor(bdata['poses'][fId:fId+1, :3]).to(comp_device) # controls the global root orientation
            pose_body = torch.Tensor(bdata['poses'][fId:fId+1, 3:66]).to(comp_device) # controls the body
            pose_hand = torch.Tensor(bdata['poses'][fId:fId+1, 66:]).to(comp_device) # controls the finger articulation
            betas = torch.Tensor(bdata['betas'][fId:fId+1]).to(comp_device) # controls the body shape
            trans = torch.Tensor(bdata['trans'][fId:fId+1]).to(comp_device)  
            # print(f"pose_body: {pose_body.shape}, pose_hand: {pose_hand.shape}, betas: {betas.shape}, root_orient: {root_orient.shape}")  
            body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient)
            joint_loc = body.Jtr[0] + trans
            pose_seq.append(joint_loc.unsqueeze(0))
    pose_seq = torch.cat(pose_seq, dim=0)
    
    pose_seq_np = pose_seq.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)
    np.save(save_path, pose_seq_np_n)

    # process obj pose data
    angle, trans = bdata_obj['angles'], bdata_obj['trans']
    rot = Rotation.from_rotvec(angle).as_matrix()
    mat = np.eye(4)[np.newaxis].repeat(rot.shape[0], axis=0)
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = trans
    trans_matrix_eye4 = np.eye(4)[np.newaxis]
    trans_matrix_eye4[0, :3, :3] = trans_matrix
    mat = trans_matrix_eye4 @ mat

    rot, trans = mat[:, :3, :3], mat[:, :3, 3]
    rot = Rotation.from_matrix(rot).as_rotvec()
    obj_pose = np.concatenate([rot, trans], axis=-1)
    
    save_path_obj = save_path.replace('smpl_fit_all.npy', 'object_fit_all.npy')
    np.save(save_path_obj, obj_pose)
    return fps

# %%
group_path = group_path
all_count = sum([len(paths) for paths in group_path])
cur_count = 0


# %%
import time
for paths in group_path:
    dataset_name = paths[0].split('/')[2]
    pbar = tqdm(paths)
    pbar.set_description('Processing: %s'%dataset_name)
    fps = 0
    for path in pbar:
        save_path = path.replace('./dataset/raw_behave', './dataset/joints_behave')
        save_path = save_path[:-3] + 'npy'
        try:
            fps = behave_to_pose(path, save_path)
        except:
            print('Error: ', path)
            continue
        
    cur_count += len(paths)
    print('Processed / All (fps %d): %d/%d'% (fps, cur_count, all_count) )
    time.sleep(0.5)



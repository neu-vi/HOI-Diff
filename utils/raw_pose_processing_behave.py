# %% [markdown]
# ## Extract Poses from Amass Dataset

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib notebook
# %matplotlib inline

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

# %% [markdown]
# ### Please remember to download the following subdataset from AMASS website: https://amass.is.tue.mpg.de/download.php. Note only download the <u>SMPL+H G</u> data.
# * ACCD (ACCD)
# * HDM05 (MPI_HDM05)
# * TCDHands (TCD_handMocap)
# * SFU (SFU)
# * BMLmovi (BMLmovi)
# * CMU (CMU)
# * Mosh (MPI_mosh)
# * EKUT (EKUT)
# * KIT  (KIT)
# * Eyes_Janpan_Dataset (Eyes_Janpan_Dataset)
# * BMLhandball (BMLhandball)
# * Transitions (Transitions_mocap)
# * PosePrior (MPI_Limits)
# * HumanEva (HumanEva)
# * SSM (SSM_synced)
# * DFaust (DFaust_67)
# * TotalCapture (TotalCapture)
# * BMLrub (BioMotionLab_NTroje)
# 
# ### Unzip all datasets. In the bracket we give the name of the unzipped file folder. Please correct yours to the given names if they are not the same.

# %% [markdown]
# ### Place all files under the directory **./amass_data/**. The directory structure shoud look like the following:  
# ./amass_data/  
# ./amass_data/ACCAD/  
# ./amass_data/BioMotionLab_NTroje/  
# ./amass_data/BMLhandball/  
# ./amass_data/BMLmovi/   
# ./amass_data/CMU/  
# ./amass_data/DFaust_67/  
# ./amass_data/EKUT/  
# ./amass_data/Eyes_Japan_Dataset/  
# ./amass_data/HumanEva/  
# ./amass_data/KIT/  
# ./amass_data/MPI_HDM05/  
# ./amass_data/MPI_Limits/  
# ./amass_data/MPI_mosh/  
# ./amass_data/SFU/  
# ./amass_data/SSM_synced/  
# ./amass_data/TCD_handMocap/  
# ./amass_data/TotalCapture/  
# ./amass_data/Transitions_mocap/  
# 
# **Please make sure the file path are correct, otherwise it can not succeed.**

# %%
# Choose the device to run the body model on.
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
from human_body_prior.body_model.body_model import BodyModel

male_bm_path = './body_models/smplh/SMPLH_MALE.npz'
male_dmpl_path = './body_models/dmpls/male/model.npz'

female_bm_path = './body_models/smplh/SMPLH_FEMALE.npz'
female_dmpl_path = './body_models/dmpls/female/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

# male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas).to(comp_device)
faces = c2c(male_bm.f)

# female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path).to(comp_device)
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
save_root = './dataset/pose_data_behave'
save_folders = [folder.replace('./dataset/raw_behave', './dataset/pose_data_behave') for folder in folders]
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
    with open(seq_info_path, "r") as f:
        seq_info = json.load(f)
    gender = seq_info["gender"]

    src_path_obj = src_path.replace('smpl_fit_all.npz', 'object_fit_all.npz')

    bdata = np.load(src_path, allow_pickle=True)
    bdata_obj = np.load(src_path_obj, allow_pickle=True)
    fps = 30
    frame_number = bdata['trans'].shape[0]

    fId = 0 # frame id of the mocap sequence
    pose_seq = []
    if gender == 'male':
        bm = male_bm
    else:
        bm = female_bm
    down_sample = int(fps / ex_fps)
#     print(frame_number)
#     print(fps)
    
    with torch.no_grad():
        for fId in range(0, frame_number, down_sample):
            root_orient = torch.Tensor(bdata['poses'][fId:fId+1, :3]).to(comp_device) # controls the global root orientation
            pose_body = torch.Tensor(bdata['poses'][fId:fId+1, 3:66]).to(comp_device) # controls the body
            pose_hand = torch.Tensor(bdata['poses'][fId:fId+1, 66:]).to(comp_device) # controls the finger articulation
            betas = torch.Tensor(bdata['betas'][fId:fId+1]).to(comp_device) # controls the body shape
            trans = torch.Tensor(bdata['trans'][fId:fId+1]).to(comp_device)    
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

# %% [markdown]
# This will take a few hours for all datasets, here we take one dataset as an example
# 
# To accelerate the process, you could run multiple scripts like this at one time.

# %%
import time
for paths in group_path:
    dataset_name = paths[0].split('/')[2]
    pbar = tqdm(paths)
    pbar.set_description('Processing: %s'%dataset_name)
    fps = 0
    for path in pbar:
        save_path = path.replace('./dataset/raw_behave', './dataset/pose_data_behave')
        save_path = save_path[:-3] + 'npy'
        try:
            fps = behave_to_pose(path, save_path)
        except:
            print('Error: ', path)
            continue
        
    cur_count += len(paths)
    print('Processed / All (fps %d): %d/%d'% (fps, cur_count, all_count) )
    time.sleep(0.5)

# %% [markdown]
# The above code will extract poses from **AMASS** dataset, and put them under directory **"./pose_data"**

# %% [markdown]
# The source data from **HumanAct12** is already included in **"./pose_data"** in this repository. You need to **unzip** it right in this folder.

# %% [markdown]
# ## Segment, Mirror and Relocate Motions

# %%
import codecs as cs
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin

def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

save_dir = './dataset/joints_behave'
total_amount = len(group_path[0])
fps = 30

for i in tqdm(range(total_amount)):
    path = group_path[0][i]
    source_path = path.replace('./dataset/raw_behave', './dataset/pose_data_behave')
    source_path = source_path[:-3] + 'npy'
    try:
        data = np.load(source_path)
    except:
        print('Error: ', source_path)
        continue
    new_name = source_path.split('/')[-2]
    # data[..., 0] *= -1
    
    # data_m = swap_left_right(data)

    source_path_obj = source_path.replace('smpl_fit_all.npy', 'object_fit_all.npy')
    data_obj = np.load(source_path_obj)

    np.save(pjoin(save_dir, new_name), data)
    # np.save(pjoin(save_dir, 'M'+new_name), data_m)





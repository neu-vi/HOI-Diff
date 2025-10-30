import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
# from human_body_prior.tools.omni_tools import copy2cpu as c2c







os.environ['PYOPENGL_PLATFORM'] = 'egl'

# %%
# Choose the device to run the body model on.
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
from human_body_prior.body_model.body_model import BodyModel
neutral_bm_path = './body_models/smplx/SMPLX_NEUTRAL.npz'


male_bm_path = './body_models/smplx/SMPLX_MALE.npz'
male_dmpl_path = './body_models/dmpls/male/model.npz'

female_bm_path = './body_models/smplx/SMPLX_FEMALE.npz'
female_dmpl_path = './body_models/dmpls/female/model.npz'


# male_bm_path = './body_models/smplh/SMPLH_MALE.npz'


num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

# male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path).to(comp_device)
male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas).to(comp_device)
# faces = c2c(male_bm.f)

# female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path).to(comp_device)
female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas).to(comp_device)


neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas).to(comp_device)







# %%
trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0]])

def smplx_to_joints(pose_data, gender):
    ex_fps = 30
    fps = 120
    frame_number = pose_data.shape[0]

    fId = 0 # frame id of the mocap sequence
    pose_seq = []
    if gender == 'male':
        bm = male_bm
    else:
        bm = female_bm

    # bm = neutral_bm
    down_sample = int(fps / ex_fps)

    
    
    with torch.no_grad():
        for fId in range(0, frame_number, 1):
            root_orient = torch.Tensor(pose_data[fId:fId+1, :3]).to(comp_device) # controls the global root orientation
            pose_body = torch.Tensor(pose_data[fId:fId+1, 3:3+63]).to(comp_device) # controls the body
            pose_hand = torch.zeros([1, 90]).to(comp_device) # controls the finger articulation
            trans = torch.Tensor(pose_data[fId:fId+1, 66:69]).to(comp_device) # controls the body shape
            betas = torch.Tensor(pose_data[fId:fId+1, 69:]).to(comp_device)    
            body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient)
            joint_loc = body.Jtr[0] + trans
            pose_seq.append(joint_loc.unsqueeze(0))
    pose_seq = torch.cat(pose_seq, dim=0)

    
    num_joints = 22
    return_joints24 = False
    # cat all genders and reorder to original batch ordering
    if return_joints24:
        x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0) # () X 52 X 3 
        lmiddle_index= 28 
        rmiddle_index = 43 
        x_pred_smpl_joints = torch.cat((x_pred_smpl_joints_all[:, :22, :], \
            x_pred_smpl_joints_all[:, lmiddle_index:lmiddle_index+1, :], \
            x_pred_smpl_joints_all[:, rmiddle_index:rmiddle_index+1, :]), dim=1) 
    else:
        pose_seq = pose_seq[:, :num_joints, :].detach().cpu().numpy()
    return pose_seq
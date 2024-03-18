import torch
from data_loaders.behave.scripts.motion_process import recover_from_ric
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from visualize.simplify_loc2rot import joints2smpl
from model.rotation2xyz import Rotation2xyz
from diffusion.losses import point2point_signed
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
from trimesh import Trimesh
import trimesh
import math



class Guide_Contact:
    def __init__(self,
                 inv_transform_th=None,
                 classifiler_scale=10.0,
                 guidance_style='xstart',
                 stop_cond_from=0,
                 use_global=False,
                 batch_size=10,
                 afford_sample=None,
                 mean=None,
                 std=None
                 ):

        self.classifiler_scale = classifiler_scale
        self.inv_transform_th = inv_transform_th 
        self.n_joints = 22
        self.sigmoid = torch.nn.Sigmoid()
        self.mean=mean
        self.std=std
        self.use_global = use_global
        self.batch_size = batch_size


        self.afford_sample = afford_sample

        self.loss_all = []


    def __call__(self, x, t, y=None, human_mean=None): # *args, **kwds):
        """
        Args:
            target: [bs, 120, 22, 3]
            target_mask: [bs, 120, 22, 3]
        """

            
        # return x.detach()
        loss, grad, loss_list = self.gradients(x, t, self.afford_sample, y['obj_points'], y['obj_normals'])

            
        return loss, grad, loss_list

    def gradients(self, x, t, afford_sample, obj_points, obj_normals):
        with torch.enable_grad():
            n_joints = 22 
            x.requires_grad_(True)

            
            sample = x.permute(0, 2, 3, 1) * torch.from_numpy(self.std).to(x.device) + torch.from_numpy(self.mean).to(x.device)


            B, _, T , _ = sample.shape

            sample_obj = sample[..., 263:]
            sample_obj = sample_obj.permute(0, 1, 3, 2)
            sample = recover_from_ric(sample.float(), n_joints)
            sample = sample[:,:,:,:n_joints*3]
            joints_output = sample.reshape(sample.shape[0], sample.shape[2], n_joints, 3)

            obj_output = sample_obj[:,0,:,:].permute(0,2,1).float()


            contact_idxs = []
            o_afford_labels = []
            for i in range(afford_sample.shape[0]):
                contact_prob = afford_sample[i,3:,0, :].permute(1,0)
                contact_pos = afford_sample[i,:3, 0, :].permute(1,0)
                contact_idx = torch.where(contact_prob>0.65)[0]
                points = obj_points[i]
                if len(contact_idx)>0:
                    sel_pos = contact_pos[contact_idx].to(points.device)                    
                    dist = torch.cdist(sel_pos, points)
                    min_dist_idx = torch.argmin(dist, dim=-1)
                    o_afford_labels.append(min_dist_idx.detach().cpu().numpy())
                    contact_idxs.append(contact_idx.detach().cpu().numpy())
                else:
                    o_afford_labels.append(np.array([-1]))
                    contact_idxs.append(np.array([-1]))

          
            batch_size = joints_output.size(0)
            all_loss_joints_contact = 0
            all_loss_object_contact = 0

        
            contact_loss= torch.zeros(0).to(x.device)


            all_loss_static = torch.zeros(0).to(x.device)
            all_loss_static_xz = torch.zeros(0).to(x.device)

            all_local_rot = torch.zeros(0).to(x.device)
            all_close_points_loss = torch.zeros(0).to(x.device)


            for i in range(B):

                              
                # center
                vertices = obj_points[i][:-2,:].float()
                center = torch.mean(vertices, 0)
                vertices -= center
                center_ = torch.mean(vertices, 0)

                init_y = center_[1:2] - vertices[:, 1].min()
            
                contact_vertices = obj_points[i][-2:,:].float()
    

                obj_normal = obj_normals[i]

                pred_angle, pred_trans = obj_output[i, :, :3].transpose(1,0), obj_output[i, :, 3:].transpose(1,0)
                pred_rot = axis_angle_to_matrix(pred_angle.transpose(1,0))

                pred_points = torch.matmul(contact_vertices.unsqueeze(0), pred_rot.permute(0, 2, 1)) + pred_trans.transpose(1, 0).unsqueeze(1)

                all_pred_points = torch.matmul(obj_points[i].float().unsqueeze(0), pred_rot.permute(0, 2, 1)) + pred_trans.transpose(1, 0).unsqueeze(1)
                
                if contact_idxs[i].any() !=-1:
                    sel_joints = np.array([0,9,10,11,16,17,20,21])
                    contact_idxs[i] = np.array([6, 7])
                    o_afford_labels[i] = o_afford_labels[i][:2]

                    sel_idx = sel_joints[contact_idxs[i]]
                    loss_contact = torch.norm((joints_output[i, :, sel_idx,:] - all_pred_points[:, o_afford_labels[i],  :]), dim=-1)
                    contact_loss = torch.cat([contact_loss, loss_contact.sum(-1).unsqueeze(0)], dim=0)




            total_loss_contact = 1.0 * contact_loss.sum()

            loss_sum = total_loss_contact
            
            self.loss_all.append(loss_sum)

            grad = torch.autograd.grad([loss_sum], [x])[0]
            x.detach()
        return loss_sum, grad, self.loss_all



def create_capsule(all_joint_pos, radius=0.12):

    all_joint_pos = all_joint_pos.detach().cpu().numpy()

    capsule_points = []
    for i in range(all_joint_pos.shape[0]):
        joint_pos = all_joint_pos[i]
        foot = joint_pos[:,1].argmin()
        head = joint_pos[:,1].argmax()
        x = joint_pos[0,0]
        y = joint_pos[0,2]

        # Parameters of the capsule
        # radius = 0.12  # Radius of the hemispheres and the cylinder
        height = joint_pos[head, 1]  # Height of the cylindrical body
        num_points_per_hemisphere = 20  # Number of points in each hemisphere
        num_points_in_cylinder = 30  # Number of points in the cylindrical body

        # Create points for the first hemisphere
        phi = np.linspace(0, np.pi, num_points_per_hemisphere)
        theta = np.linspace(0, 2 * np.pi, num_points_per_hemisphere)
        phi, theta = np.meshgrid(phi, theta)

        phi_cylinder = np.linspace(0, 2 * np.pi, num_points_in_cylinder)
        y_cylinder = np.linspace(0, height, num_points_in_cylinder)
        phi_cylinder, y_cylinder = np.meshgrid(phi_cylinder, y_cylinder)
        x_cylinder = joint_pos[0,0] +radius * np.cos(phi_cylinder)
        z_cylinder = joint_pos[0,2] + radius * np.sin(phi_cylinder)

        # Combine the points from both hemispheres and the cylinder
        x = x_cylinder.ravel()[:,np.newaxis]
        y = y_cylinder.ravel()[:,np.newaxis]
        z = z_cylinder.ravel()[:,np.newaxis]

        xyz = np.concatenate([x,y,z],-1)

        capsule_points.append(xyz)

    capsule_points = np.array(capsule_points)
    # print(f"xx {capsule_points.shape}")

    return torch.from_numpy(capsule_points)



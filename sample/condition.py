import torch
from data_loaders.behave.scripts.motion_process import recover_from_ric
import torch.nn.functional as F
import numpy as np
import os
# import chamfer_pytorch.dist_chamfer as ext
import matplotlib.pyplot as plt
from visualize.simplify_loc2rot import joints2smpl
from model.rotation2xyz import Rotation2xyz
from diffusion.losses import point2point_signed
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
from sample.tools import *
# import open3d as o3d
from trimesh import Trimesh
import trimesh
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import math


# simplified_mesh = {
#     "backpack":"backpack/backpack_f1000.ply",
#     'basketball':"basketball/basketball_f1000.ply",
#     'boxlarge':"boxlarge/boxlarge_f1000.ply",
#     'boxtiny':"boxtiny/boxtiny_f1000.ply",
#     'boxlong':"boxlong/boxlong_f1000.ply",
#     'boxsmall':"boxsmall/boxsmall_f1000.ply",
#     'boxmedium':"boxmedium/boxmedium_f1000.ply",
#     'chairblack': "chairblack/chairblack_f2500.ply",
#     'chairwood': "chairwood/chairwood_f2500.ply",
#     'monitor': "monitor/monitor_closed_f1000.ply",
#     'keyboard':"keyboard/keyboard_f1000.ply",
#     'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
#     'stool':"stool/stool_f1000.ply",
#     'tablesquare':"tablesquare/tablesquare_f2000.ply",
#     'toolbox':"toolbox/toolbox_f1000.ply",
#     "suitcase":"suitcase/suitcase_f1000.ply",
#     'tablesmall':"tablesmall/tablesmall_f1000.ply",
#     'yogamat': "yogamat/yogamat_f1000.ply",
#     'yogaball':"yogaball/yogaball_f1000.ply",
#     'trashbin':"trashbin/trashbin_f1000.ply",
# }

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
        # self.points = points
        self.classifiler_scale = classifiler_scale
        self.inv_transform_th = inv_transform_th 
        self.n_joints = 22
        self.sigmoid = torch.nn.Sigmoid()
        self.mean=mean
        self.std=std
        self.use_global = use_global
        self.batch_size = batch_size
        # self.init_rot = torch.tensor([0.61041163,  -1.14469556, -1.26324003])

        # self.init_pos = torch.tensor([0.17,  0.01, 0.39])
        # self.is_static = np.array([0,0,1,1,0,0])
        self.ground_pose = np.zeros([batch_size, 6])
        self.top_axis = np.zeros([batch_size, 2])
        self.fix_rot =  torch.tensor(np.zeros([batch_size, 3])).float().to('cuda')

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
            if self.use_global:
                sample_obj = sample[..., 262:]
                sample_obj = sample_obj.permute(0, 1, 3, 2)
                joints_output = sample[..., :66].reshape(B, T, n_joints, 3)
            else:
                sample_obj = sample[..., 263:]
                sample_obj = sample_obj.permute(0, 1, 3, 2)
                sample = recover_from_ric(sample.float(), n_joints)
                sample = sample[:,:,:,:n_joints*3]
                joints_output = sample.reshape(sample.shape[0], sample.shape[2], n_joints, 3)

            obj_output = sample_obj[:,0,:,:].permute(0,2,1).float()

            # print(f"=========  conobj_outputtact_labels  :{obj_output.shape}")


            contact_idxs = []
            o_afford_labels = []
            for i in range(afford_sample.shape[0]):
                contact_prob = afford_sample[i,3:,0, :].permute(1,0)
                contact_pos = afford_sample[i,:3, 0, :].permute(1,0)
            # sel_joints = [0,9,10,11,16,17,20,21]
            # h_afford_labels = torch.zeros([contact_prob.shape[0], 22]).to(dist_util.dev())
            # for i in range(afford_sample.shape[0]):
            #     h_afford_labels[i, sel_joints] = contact_prob[i]
            # h_mask = (h_afford_labels>0.6).int()
                contact_idx = torch.where(contact_prob>0.65)[0]
                points = obj_points[i]
                if len(contact_idx)>0:
                    sel_pos = contact_pos[contact_idx].to(points.device)                    
                    dist = torch.cdist(sel_pos, points)
                    min_dist_idx = torch.argmin(dist, dim=-1)
                    o_afford_labels.append(min_dist_idx.detach().cpu().numpy())
                    contact_idxs.append(contact_idx.detach().cpu().numpy())
                    # print(f"======= {o_afford_labels}")
                else:
                    o_afford_labels.append(np.array([-1]))
                    contact_idxs.append(np.array([-1]))

            # print(f"=========  o_afford_labels  :{o_afford_labels}  contact_idxs : {contact_idxs}")

            # ind , is_static =  contact_labels[:,:2], contact_labels[:,2:]
    
            # o_contact_labels = contact_labels[:,22:].float()
          
            batch_size = joints_output.size(0)
            all_loss_joints_contact = 0
            all_loss_object_contact = 0
            all_loss_h_collision = 0
            all_loss_o_collision = 0

        
            h_contact_dist = torch.zeros(0).to(x.device)
            o_contact_dist = torch.zeros(0).to(x.device)

            all_loss_static = torch.zeros(0).to(x.device)
            all_loss_static_xz = torch.zeros(0).to(x.device)

            all_local_rot = torch.zeros(0).to(x.device)
            all_close_points_loss = torch.zeros(0).to(x.device)

            collisione_loss = 0.0


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
                    # print(f"======= {contact_idxs[i]}")

                    sel_idx = sel_joints[contact_idxs[i]]
                    loss_contact = torch.norm((joints_output[i, :, sel_idx,:] - all_pred_points[:, o_afford_labels[i],  :]), dim=-1)

                # if ind[i, 0] != ind[i, 1]:
                #     loss_contact1 = torch.norm((joints_output[i, :, ind[i, 0:1]] - pred_points[:,  0:1]), dim=-1) 
                #     loss_contact2 = torch.norm((joints_output[i, :, ind[i, 1:2]] - pred_points[:,  1:2]), dim=-1)                   
                #     loss_contact_h = loss_contact1  + loss_contact2
                # else:
                #     loss_contact_h = torch.norm((joints_output[i, :, ind[i, 0:1]] - pred_points[:, -2:-1]), dim=-1)

                    h_contact_dist = torch.cat([h_contact_dist, loss_contact.sum(-1).unsqueeze(0)], dim=0)





                
                # if is_static[i]==1:
                # if i == 0:

                #     if t[0] == 999:
                #         # Define the initial pose of the object (position and orientation)
                #         # initial_x = obj_output[i, 0, 3].detach().cpu().numpy()   # Replace with the actual initial x-coordinate
                #         initial_y = init_y.detach().cpu().numpy() # Replace with the actual initial y-coordinate
                #         # initial_z = obj_output[i, 0, 5].detach().cpu().numpy()    # Replace with the actual initial z-coordinate
                        
                #         # Define the desired orientation for the object to be horizontal (roll, pitch, yaw)
                #         desired_roll = 0.0
                #         desired_pitch = obj_output[i, 0, 1]
                #         desired_yaw = 0.0  # Keep the initial yaw angle
                #         # Apply the rotation to the initial orientation
                #         new_roll, new_pitch, new_yaw = desired_roll, desired_pitch, desired_yaw

                #         new_rot = np.array([desired_roll, desired_pitch.detach().cpu().numpy(), desired_yaw])

                #         self.ground_pose[i, :3] = new_rot
                #         self.ground_pose[i, 4] = initial_y



                    # ground_target_rot = torch.tensor(self.ground_pose[i, :3]).unsqueeze(0).to(x.device)
                    # ground_target_trans = torch.tensor(self.ground_pose[i, 3:]).unsqueeze(0).to(x.device)

                    # loss_static_rot = torch.norm((obj_output[i, :, [0,2]] -  ground_target_rot[:, [0,2]]), dim=-1)
                    # loss_static_y =  torch.norm((obj_output[i, :, 3:] - ground_target_trans), dim=-1)
                    # # loss_static_rot = torch.norm((obj_output[i, :, :3] -  ground_target_rot), dim=-1) + torch.norm((obj_output[i, :, :3] - torch.mean(obj_output[i, :, :3], dim=1, keepdim=True)), dim=-1)# handle xyz-rotation
                    # # loss_static_y =  torch.norm((obj_output[i, :, 3:] - ground_target_trans), dim=-1) +  torch.norm((obj_output[i, :, [3,5]] - torch.mean(obj_output[i, :, [3,5]], dim=1, keepdim=True)), dim=-1)
                    # # if t[0] < 1000 and t[0] > 700:
                    # #     loss_static_rot = torch.norm((obj_output[i, :, [0, 2]] - ground_target_rot[:,[0, 2]]), dim=-1) # handle xz-rotation
                    # #     loss_static_y = torch.norm((obj_output[i, :, 3:] - ground_target_trans), dim=-1)  # handle xyz_pos
                    # # else:
                    # #     # fix object y-axis rotation at that time
                    # #     loss_static_rot = torch.norm((obj_output[i, :, :3] - ground_target_rot), dim=-1) # handle xyz-rotation
                    # #     loss_static_y = torch.norm((obj_output[i, :, 3:] - ground_target_trans), dim=-1) + torch.norm((obj_output[i, :, [1]] - torch.mean(obj_output[i, :, [1]], dim=1, keepdim=True)), dim=-1)   # handle xyz_pos

                    #     # loss_static_rot = torch.norm((obj_output[i, :, :3] -  torch.mean(obj_output[i, :, :3], dim=1, keepdim=True)), dim=-1) # handle xyz-rotation
                    #     # loss_static_y = torch.norm((obj_output[i, :, 3:] - torch.mean(obj_output[i, :, 3:], dim=1, keepdim=True)), dim=-1) 

                    # # loss_statix_xz = torch.norm((obj_output[i, 1:, [3,5]] - obj_output[i, :-1, [3,5]]), dim=-1)
                    # loss_static = loss_static_rot + loss_static_y 
                    # all_loss_static = torch.cat([all_loss_static, loss_static.unsqueeze(0)], dim=0) 
                    # # all_loss_static_xz = torch.cat([all_loss_static_xz, loss_statix_xz.unsqueeze(0)], dim=0) 
                
                # if t[0] == 700:
                #     local_x_axis =  obj_output[i, :, 0]
                #     local_y_axis =  obj_output[i, :, 1]
                #     local_z_axis =  obj_output[i, :, 2]
                #     local_axis = torch.stack((local_x_axis, local_y_axis, local_z_axis), dim=1)
                #     _, top_axis = torch.var(local_axis, dim=0).topk(2,  largest=True)
                #     # _, top_axis = torch.mean(local_axis, dim=0).topk(2,  largest=False)
                #     top_axis, _ = torch.sort(top_axis)
                #     print(f"================ {top_axis}")

                #     self.top_axis[i] = top_axis.cpu().numpy()
                    # self.top_axis=[0]
                    # self.fix_rot[i, self.top_axis] =  torch.mean(obj_output[i, :, self.top_axis], dim=0, keepdim=True)
                    
        
                # if ind[i, 0] != ind[i, 1]:
                #     # loss_smooth_obj_rot_mean =  F.mse_loss(obj_output[i, :,  :], torch.mean(obj_output[i, :, :], dim=0, keepdim=True)) 
                #     loss_smooth_obj_rot_mean =   F.mse_loss(obj_output[i, :, [0,  2]], torch.mean(obj_output[i, :, [0,  2]], dim=1, keepdim=True)) * 100
                #     all_local_rot = torch.cat([all_local_rot, loss_smooth_obj_rot_mean.unsqueeze(0)], dim=0)

                # # #################################################
                # contact map loss
                # if t[0] < 1000:
                    # capsule_points = create_capsule(joints_output[i], radius=0.25)
                    # capsule_points = capsule_points.float().to(x.device)

                    # obj_normal_ = vertex_normals(pred_points, obj_normal.unsqueeze(0).repeat(T, 1, 1))

                    # # print(f"test: {type(obj_normal)}  {type(capsule_points)} ")

                    # o2h_signed_skel, h2o_signed_skel, o2h_idx, h2o_idx, o2h, h2o = point2point_signed(capsule_points, pred_points, y_normals=obj_normal_, return_vector=True)

                    # # w_dist_neg = (o2h_signed_skel < 0).view(T, -1).float()
                    # # penetrate = w_dist_neg.mean(dim=-1).mean(dim=0)

                    # w_dist_neg2 = (h2o_signed_skel < 0).view(T, -1).float()
                    # penetrate2 = w_dist_neg2.mean(dim=-1).mean(dim=0)
                    

                    # collisione_loss += penetrate2

                    # all_idx = np.arange(22)
                    # ignore_idx = np.setdiff1d(all_idx, ind[i].cpu().numpy())

                    # distances = torch.cdist(joints_output[i,:, ignore_idx], all_pred_points)  # Using Euclidean distance

                    # threshold_distance = 0.05
            
                    # # Apply penalty for points closer than the threshold distance
                    # close_points_loss = torch.mean(torch.relu(threshold_distance - distances))
                    # # print(f"==== {all_pred_points.shape}")
                    # smooth_obj_points_speed = F.mse_loss(all_pred_points[1:], all_pred_points[:-1]) * 100

                    # all_close_points_loss = torch.cat([all_close_points_loss, smooth_obj_points_speed.unsqueeze(0)], dim=0) 
  

                    # plot('./test_{}.mp4'.format(i), torch.cat([joints_output.squeeze()[i], pred_points, capsule_points], dim=1).detach().cpu().numpy(), None, None, "test", fps=20)
            # for i in range(B):

            #    if t[0] < 700:    
            #         loss_smooth_obj_rot_mean = F.mse_loss(obj_output[i, :,  self.top_axis[i]],  obj_output[i, 0:1,  self.top_axis[i]]) 
            #         loss_smooth_obj_rot_mean = F.mse_loss(obj_output[i, :,  self.top_axis[i]],  self.fix_rot[i,  self.top_axis[i]].unsqueeze(0)) * 800
            #         # if ind[i, 0] == ind[i, 1]:
            #         # loss_smooth_obj_rot_mean =  F.mse_loss(obj_output[i, :,  [0,2]], obj_output[i, :1, [0,2]]) * 500
            #         # loss_smooth_obj_rot_mean =  F.mse_loss(obj_output[i, :,  [1, 2]], torch.mean(obj_output[i, :,  [1, 2]], dim=0, keepdim=True)) * 500
            #         all_local_rot = torch.cat([all_local_rot, loss_smooth_obj_rot_mean.unsqueeze(0)], dim=0)


            # loss_smooth_obj_rot_xyz = F.mse_loss(obj_output[:, 1:, :], obj_output[:, :-1, :]) * 500.0
            

            # loss_smooth_obj_rot_mean = F.mse_loss(obj_output[:, :, [0,  2]], obj_output[:, 0:1, [0,  2]]) 

            loss_smooth_obj_rot_mean = F.mse_loss(obj_output[:, :, [0,  2]], torch.mean(obj_output[:, :, [0,  2]], dim=1, keepdim=True)) * 200
            loss_smooth_obj_rot_speed = F.mse_loss(obj_output[:, 1:, [0, 1, 2]], obj_output[:, :-1, [0, 1, 2]]) * 500
            # loss_smooth_obj_rot_y =  F.mse_loss(obj_output[:, 1:, [1]], torch.mean(obj_output[:, :, [1]], dim=1, keepdim=True)) * 500
            # loss_smooth_obj_rot_xz =  F.mse_loss(obj_output[:, :, [0, 2]], torch.mean(obj_output[:, :, [0, 2]], dim=1, keepdim=True)) * 1.0

            # loss_smooth_obj_local_rot =  F.mse_loss(all_local_rot[:, :, :], torch.mean(all_local_rot[:, :, :], dim=1, keepdim=True)) * 1000
            # loss_smooth_obj_local_rot =  F.mse_loss(all_local_rot[:, 1:, :], all_local_rot[:, :-1, :]) * 1000
            # loss_smooth_obj_trans = F.mse_loss(obj_output[:, 1:, 3:], obj_output[:, :-1, 3:]) * 500.0






            loss_contact = 1.0 * h_contact_dist.sum()

            

            loss_static =   1.0 * all_loss_static.sum()

                        
            # loss_collision = collisione_loss * 100

            loss_smooth_obj_rot  = loss_smooth_obj_rot_mean



            # loss_smooth_obj_rot = all_local_rot.sum() 

            # loss_close_points_loss = all_close_points_loss.sum()

            
            # if t[0] < 700:
            #     loss_sum = loss_contact  + loss_static
            # else:
            loss_sum = loss_contact + loss_smooth_obj_rot
            
            if t[0] == 999:
                print("============= Init Loss in Guidance ==============")
                # print(f"ind  {ind}")
                # print(f" Loss_Contact: {loss_contact} Loss_Smooth_obj_rot:{loss_smooth_obj_rot} Loss_Static {loss_static}   loss_collision {loss_smooth_obj_rot_speed}")
            if t[0] == 99:
                print("============= Middle Loss in Guidance ==============")
                print(f" Loss_Contact: {loss_contact}  Loss_Smooth_obj_rot:{loss_smooth_obj_rot} ")
            if t[0] == 1:
                # print("============= Final Loss in Guidance ==============")
                # print(f" Is_static {is_static}")
                print(f" Loss_Contact: {loss_contact}  Loss_Smooth_obj_rot:{loss_smooth_obj_rot} ")
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



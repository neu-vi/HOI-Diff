from model.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl
from scipy.spatial.transform import Rotation


class npy2obj:
    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True, if_color=False):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)


        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            self.motions['motion'] = motion_tensor.cpu().numpy()
        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.real_num_frames = self.motions['lengths'][self.absl_idx]



        self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=False)
        self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)
        self.vertices += self.root_loc
        # put one the floor y = 0
        floor_height = self.vertices[0].min(0)[0].min(-1)[0][1]
        self.vertices[:, :, 1] -= floor_height

        if if_color and 'h_contact' in self.motions.keys():
        #     self.colors = np.load('./gt_files/hc_Date01_Sub01_backpack_back.npy'.format(self.motions['obj_name'][0]))
        #     self.colors = np.stack([self.colors, self.colors, self.colors, np.ones_like(self.colors)], axis=1).astype(np.int8) * 255
        # else:
            posa_path= '/work/vig/yimingx/POSA/mesh_ds/downsample.npy'
            choose = np.load(posa_path)
            self.contact_idxs = np.zeros((self.bs, self.vertices.shape[1], self.nframes), dtype=np.int8)
            self.contact_idxs[:, choose, :] = self.motions['h_contact'][self.absl_idx].astype(np.int8)

            self.colors = np.zeros((self.bs, self.vertices.shape[1], 4, self.nframes), dtype=np.int8)
            colors = self.motions['h_contact'][self.absl_idx].astype(np.int8) * 255
            colors = np.stack([colors, colors, colors, np.ones_like(colors)], axis=2)
            self.colors[:, choose, :, :] = colors
        else:
            self.colors = None 
        
    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()
    
    def get_colors(self, sample_i, frame_i):
        if self.colors is None:
            return None
        else:
            return self.colors[sample_i, ..., frame_i].tolist()

    def get_trimesh(self, sample_i, frame_i):
        self.get_vertices(sample_i, frame_i)
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces,
                       vertex_colors=self.get_colors(sample_i, frame_i))

    def save_obj(self, save_path,  frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_ply(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        mesh.export(save_path)
        return save_path
    
    def save_npy(self, save_path):
        # print(f"motion: { type(self.motions['motion'])} frame : {self.real_num_frames}  vet {type(self.vertices.detach().cpu().numpy())}")
        data_dict = {
            'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
            'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
            'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0, :, :, :self.real_num_frames].detach().cpu().numpy(),
            # 'contact_idx': self.contact_idxs[0, :, :self.real_num_frames],
            'contact_idx': None,
            'text': self.motions['text'][0],
            'length': self.real_num_frames,
        }
        np.save(save_path, data_dict)


        # # Conversion of data to .pkl file for 'softcat477/SMPL-to-FBX/' 
        # # Change 'FbxTime.eFrames60's in SMPL-to-FBX/FbxReadWriter.py to 'FbxTime.eFrames30'
        # # Known issues: Glitches below 30 fps exports in 'softcat477/SMPL-to-FBX/' (20 fps expected)
        # import utils.rotation_conversions as geometry
        # i = range(self.real_num_frames)
        # poses_raw = []
        # latest = None
        # for idx, x in np.ndenumerate(np.array(torch.flatten(geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(torch.tensor(self.motions['motion'][0, :-1, :, i])))))):
        #     poses_raw.append(x)
        #     latest = x
        # smpl_poses = np.array(poses_raw).reshape(self.real_num_frames, 72)
        # trans_raw = []
        # latest = None
        # for idx, x in np.ndenumerate(self.motions['motion'][0, -1, :3, i]):   
        #     trans_raw.append(x)
        #     latest = x
        # smpl_trans = np.array(trans_raw).reshape(self.real_num_frames, 3)*np.array([100, 1, 100])     

        # data_dict2 = {'smpl_poses': smpl_poses,'smpl_trans': smpl_trans,}
        # import pickle
        # with open(save_path +".pkl", 'wb') as pickle_file:
        #     pickle.dump(data_dict2, pickle_file)     
        # # End of Conversion of data to .pkl file for 'softcat477/SMPL-to-FBX/'


# path to the simplified mesh used for registration
simplified_mesh = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
    'clothesstand':"clothesstand_cleaned_simplified.obj",
    'floorlamp':"floorlamp_cleaned_simplified.obj",
    'tripod':"tripod_cleaned_simplified.obj",
    'whitechair':"whitechair_cleaned_simplified.obj",
    'woodchair':"woodchair_cleaned_simplified.obj"
}


class npy2obj_object:
    def __init__(self, npy_path, obj_path, sample_idx, rep_idx, device=0, cuda=True, if_color=False):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.bs, _, self.nfeats, self.nframes = self.motions['motion_obj'].shape

        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        self.num_frames = self.motions['motion_obj'][self.absl_idx].shape[-1]

        if len( self.motions['obj_name'][0].split('_'))>2:
            obj_name = [b.split('_')[2] for b in self.motions['obj_name']]
        else:
            obj_name = [b for b in self.motions['obj_name']]
        self.motions['motion_obj'] = self.motions['motion_obj']
        self.vertices, self.faces = self.pose2mesh(self.motions['motion_obj'], obj_path, obj_name)
        # self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
    
        self.real_num_frames = self.motions['lengths'][self.absl_idx]


        if if_color and 'o_contact' in self.motions.keys():
        #     self.colors = np.load('./gt_files/oc_Date01_Sub01_backpack_back.npy'.format(self.motions['obj_name'][0]))
        #     self.colors = np.stack([self.colors, self.colors, self.colors, np.ones_like(self.colors)], axis=1).astype(np.int8) * 255
        # else:
            # assume batchsize = 1
            # TODO  support any batch_size 
            obj_sample_path = '/work/vig/yimingx/behave_obj_sample/{}.npy'.format(self.motions['obj_name'][self.absl_idx])
            choose = np.load(obj_sample_path)

            self.contact_idxs = np.zeros((self.bs, self.vertices[self.absl_idx].shape[0], self.nframes), dtype=np.int8)
            self.contact_idxs[:, choose, :] = self.motions['o_contact'][self.absl_idx].astype(np.int8)

            self.colors = np.zeros((self.bs, self.vertices[self.absl_idx].shape[0], 4, self.nframes), dtype=np.int8)
            colors = self.motions['o_contact'][self.absl_idx].astype(np.int8) * 255
            colors = np.stack([colors, colors, colors, np.ones_like(colors)], axis=2)
            self.colors[:, choose, :, :] = colors
        else:
            self.colors = None

    def pose2mesh(self, motion_obj, obj_path, obj_name):
        vertices_list = []
        faces_list = []
        for b in range(self.bs):
            mesh_path = os.path.join(obj_path, simplified_mesh[obj_name[b]])
            temp_simp = trimesh.load(mesh_path)
            # vertices = temp_simp.vertices * 0.16
            vertices = temp_simp.vertices
            faces = temp_simp.faces
            # center the meshes
            center = np.mean(vertices, 0)
            vertices -= center
            # transform
            angle, trans = motion_obj[b, 0, :3], motion_obj[b, 0, 3:]
            rot = Rotation.from_rotvec(angle.transpose(1, 0)).as_matrix()
            vertices = np.matmul(vertices[np.newaxis], rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + trans.transpose(1, 0)[:, np.newaxis]
            vertices = vertices.transpose(1, 2, 0) 
            vertices_list.append(vertices)
            faces_list.append(faces)
        # return np.stack(vertices_list), np.stack(faces_list)
        return vertices_list, faces_list   #  support any batch_size


    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i][:, :, frame_i].squeeze().tolist()

    def get_faces(self, sample_i):
        return self.faces[sample_i][ :, :].squeeze().tolist()
    
    def get_colors(self, sample_i, frame_i):
        if self.colors is None:
            return None
        else:
            return self.colors[sample_i, ..., frame_i].tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.get_faces(sample_i),
                       vertex_colors=self.get_colors(sample_i, frame_i))

    def save_obj(self, save_path, sample_i, frame_i):
        mesh = self.get_trimesh(sample_i, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_ply(self, save_path, sample_i, frame_i):
        mesh = self.get_trimesh(sample_i, frame_i)
        mesh.export(save_path)
        return save_path
    
    def save_npy(self, save_path):
        data_dict = {
            'motion': self.motions['motion_obj'][self.absl_idx, :, :, :self.real_num_frames],
            'faces': np.array(self.faces[self.absl_idx]),
            'vertices': np.array(self.vertices[self.absl_idx][:, :, :self.real_num_frames]),
            # 'contact_idx': self.contact_idxs[0, :, :self.real_num_frames],
            'contact_idx': None,
            'text': self.motions['text'][self.absl_idx],
            'length': self.real_num_frames,
        }
        np.save(save_path, data_dict)

import torch
from torch.utils import data
import numpy as np
import json
import sys
sys.path.append('./')
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
import argparse
from torch.utils.data._utils.collate import default_collate
from data_loaders.utils.word_vectorizer import WordVectorizer
import trimesh
from scipy.spatial.transform import Rotation
from data_loaders.scripts.motion_process import recover_from_ric, extract_features
import scipy.sparse
from data_loaders.utils.paramUtil import *
from data_loaders.utils.plot_script import plot_3d_motion

def motion_to_rel_data(motion, is_norm=False):

    motion_bu = motion.detach().clone()
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    sample_rel_np_list = []
    for ii in range(len(motion)):
        # Data need to be [120 (timestep), 22, 3] to get feature
        sample_rel = extract_features(
            motion[ii].detach().cpu().clone().permute(2, 0,
                                                        1).cpu().numpy(),
            0.002, self.n_raw_offsets, self.kinematic_chain,
            face_joint_indx, fid_r, fid_l)
        # Duplicate last motion step to match the size
        sample_rel = torch.from_numpy(sample_rel).unsqueeze(0).float()
        # sample_rel = torch.cat(
        #     [sample_rel, sample_rel[0:1, -1:, :].clone()], dim=1)
        
        # Normalize with relative normalization
        if is_norm:
            sample_rel = (sample_rel - self.mean_rel[:263]) / self.std_rel[:263]
        sample_rel = sample_rel.unsqueeze(1).permute(0, 3, 1, 2)
        sample_rel = sample_rel.to(motion.device)
        sample_rel_np_list.append(sample_rel)

    processed_data = torch.cat(sample_rel_np_list, axis=0)

    n_joints = 22
    return processed_data

def text_to_object(text):
    obj_list = ['backpack','basketball','boxlarge','boxtiny','boxlong','boxsmall','boxmedium','chairblack','chairwood',
        'monitor','keyboard','plasticcontainer','stool','tablesquare','toolbox','suitcase','tablesmall','yogamat','yogaball','trashbin', 'clothesstand', 'floorlamp', 'tripod']

    all_obj_points = []
    all_obj_normals = []
    all_obj_names = []
    import re
    for i in range(len(text)):

        for j in range(len(obj_list)):
            if obj_list[j] in text[i]:
                name = obj_list[j]
                break
        
        # load obj points----------------
        obj_path = './dataset/behave_t2m/object_mesh'
        obj_name = name
        mesh_path = os.path.join(obj_path, obj_name + '.obj')

        temp_simp = trimesh.load(mesh_path)
        obj_points = np.array(temp_simp.vertices).astype(np.float32)
        obj_faces = np.array(temp_simp.faces).astype(np.float32)
        obj_normals = obj_faces


        # sample object points
        # obj_sample_path = './dataset/behave_t2m/object_sample/{}.npy'.format(name)
        # choose = np.load(obj_sample_path)

        obj_points = np.load(os.path.join(obj_path, 'downsample_points.npz'), allow_pickle=True)[obj_name].astype(np.float32)
        # choose = np.load(os.path.join(obj_path, 'downsample_index.npz'), allow_pickle=True)[obj_name].astype(np.float32)
        # choose = choose.astype(np.int32)
        obj_bps = np.load(pjoin(obj_path, obj_name +'_bps.npy')).astype(np.float32)
        
                
        # obj_points = obj_points[choose] 
        obj_normals = obj_bps


        all_obj_points.append(obj_points)
        all_obj_normals.append(obj_normals)
        all_obj_names.append(obj_name)

    return np.array(all_obj_points),  np.array(all_obj_normals),  np.array(all_obj_names)





def sample_to_motion(sample_global, dataset, model):
    n_joints = 22
    # (bs, 262, 1, 120)
    # In case of random projection, this already includes undoing the random projection
    
    sample = dataset.t2m_dataset.inv_transform(sample_global.cpu().permute(
        0, 2, 3, 1)).float()


    B, _, T , F = sample.shape
    sample = sample[..., :66].reshape(B, T, n_joints, 3).permute(0,2,3,1)

    return sample

def global3d_to_rel(sample_global, dataset, model, is_norm=True):
    '''We want to change the first 3 values from absolute to relative
    sample_abs shape [bs, 263, 1, 196]   [bs, 1,193, 263]
    '''
    n_joints = 22
    # (bs, 263, 1, 120)
    # In case of random projection, this already includes undoing the random projection
    sample = dataset.t2m_dataset.inv_transform(sample_global.cpu().permute(
        0, 2, 3, 1)).float()

    B, _, T , F = sample.shape
    sample_human = sample[..., :66].reshape(B, T, n_joints, 3).permute(0,2,3,1)

    # Now convert skeleton back to sample with relative representation
    sample_rel = dataset.motion_to_rel_data(sample_human, model, is_norm=is_norm)
    sample_obj = sample.permute(0, 3, 1, 2)[:, -6:, :, :-1]

    return sample_rel, sample_obj



def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

'''For use of training text motion matching model, and evaluations'''
class Text2AffordDataset(data.Dataset):
    def __init__(self, opt, split='train'):
        self.opt = opt
        self.pointer = 0
        self.dataset = opt.dataset
        self.split = split
        self.data_dir = pjoin(self.opt.data_root, self.dataset + '_t2m')
        self.max_text_len = opt.max_text_len

        data_dict = {}
        with open(pjoin(self.opt.data_root, 'dataset_split.json'), 'r') as f:
            self.split_file = json.load(f)


        cur_id_list = self.split_file[self.dataset][self.split]

        new_name_list = []
        length_list = []
        for name in tqdm(cur_id_list):
            try:
      
                # load obj points----------------
                with np.load(pjoin(self.data_dir, 'sequences/{}/object_motion.npz'.format(name))) as data:
                    obj_name = data['name']


                affordance_data = np.load(pjoin(self.data_dir, 'affordance/{}.npy'.format(name)))
                sel_joints = [0,9,10,11,16,17,20,21]
                affordance_data = affordance_data[sel_joints]  # keep the data for 8 core body joints

                obj_points = np.load(pjoin(self.data_dir, 'object_mesh/{}/sample_points.npy'.format(obj_name)))
                obj_points = np.array(obj_points).astype(np.float32)
        
                text_data = []
                flag = False
                with cs.open(pjoin(self.data_dir, 'texts/{}.txt'.format(name))) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        # f_tag = float(line_split[2])
                        # to_tag = float(line_split[3])
                        # f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        # to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        # TODO: hardcode
                        f_tag = to_tag = 0.0

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'affordance_data': affordance_data,
                                        'text': text_data,
                                        'seq_name': name,
                                        'obj_points': obj_points
                                        }

                    new_name_list.append(name)
                    # length_list.append(len(motion))
                    # TODO: harcode
                    # for i in range(1000):
                    #     data_dict[name+"_{}".format(i)] = data_dict[name]
                    #     new_name_list.append(name+"_{}".format(i))
                    #     length_list.append(len(motion))
            except Exception as err:
                print(err.__class__.__name__) # 
                print(err) 

                pass

        name_list = sorted(new_name_list, key=lambda x: x[1])

        self.data_dict = data_dict
        self.name_list = name_list


    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        affordance_data, text_list, seq_name, obj_points = data['affordance_data'],  data['text'], data['seq_name'], data['obj_points']
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, affordance_data, '_'.join(tokens), seq_name, obj_points




'''For use of training text motion matching model, and evaluations'''
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt,  split='train', w_vectorizer=None):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 30
        self.w_vectorizer = WordVectorizer(pjoin('./', 'glove'), 'our_vab')
        self.max_text_len = opt.max_text_len
        
        self.dataset = opt.dataset
        self.split = split
        self.data_dir = pjoin(self.opt.data_root, self.dataset + '_t2m')

        data_dict = {}
        with open(pjoin(self.opt.data_root, 'dataset_split.json'), 'r') as f:
            self.split_file = json.load(f)


        cur_id_list = self.split_file[self.dataset][self.split]

        data_dict = {}
  
        # id_list = id_list[:200]
        new_name_list = []
        length_list = []
        for name in tqdm(cur_id_list):
            try:
                # load hoi motion----------------
                motion = np.load(pjoin(self.data_dir, 'sequences_263_rep/{}/hoi_motion.npy'.format(name)))
                
                # load obj points----------------
                with np.load(pjoin(self.data_dir, 'sequences/{}/object_motion.npz'.format(name))) as data:
                    obj_name = data['name']


                affordance_data = np.load(pjoin(self.data_dir, 'affordance/{}.npy'.format(name)))

                # load obj points----------------
                obj_points = np.load(pjoin(self.data_dir, 'object_mesh/{}/sample_points.npy'.format(obj_name)))
                obj_points = np.array(obj_points).astype(np.float32)
                obj_bps = np.load(pjoin(self.data_dir, 'object_mesh/{}/bps_1024.npy'.format(obj_name)))
  
                        

                # TODO: hardcode
                motion = motion[:self.max_motion_length].astype(np.float32)


                if (len(motion)) < min_motion_len:
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.data_dir, 'texts/{}.txt'.format(name))) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                # new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
        
                                data_dict[name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict],
                                                    'seq_name': name,
                                                    'obj_points': obj_points,
                                                    'obj_bps':obj_bps
                                                    # 'gt_afford_labels': contact_input
                                                }
                                new_name_list.append(name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data,
                                        'seq_name': name,
                                        'obj_points': obj_points,
                                        'obj_bps':obj_bps,
                                        # # 'gt_afford_labels':contact_input
                                    }

                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as err:
                print(err.__class__.__name__) 
                print(err) 
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = np.load(pjoin(self.data_dir, 'Mean_local.npy'))
        self.std = np.load(pjoin(self.data_dir, 'Std_local.npy'))
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        data = data * self.std + self.mean
        return data

    def inv_transform_th(self, data):
        data = data * torch.from_numpy(self.std).to(
            data.device) + torch.from_numpy(self.mean).to(data.device)
        return data


    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, seq_name, obj_points, obj_bps = data['motion'], data['length'], data['text'], data['seq_name'],  data['obj_points'], data['obj_bps']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']


        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:

            if len(token.split('/'))<2:
                print(f" {seq_name}   {tokens}")
                break
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        

        "Z Normalization"
        motion = np.copy(motion)
        motion = (motion- self.mean) / self.std


        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)


        # Contact labels here for evaluation!
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), seq_name, obj_points, obj_bps





if __name__ == '__main__':

    from utils.parser_util import train_args
    import sys
    
    # 如果没有提供命令行参数，添加默认参数用于测试
    if len(sys.argv) == 1:
        sys.argv.extend(['--save_dir', './test_output'])
    
    opt = train_args()


    dataset = Text2AffordDataset(opt=opt, split='train')


    

    motion_dataset = Text2MotionDatasetV2(opt=opt, split='train')

    from torch.utils.data import DataLoader
    from data_loaders.tensors import t2m_contact_collate, t2hoi_collate


    data_loader =torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=False,
            shuffle=False,
            drop_last=True,
            collate_fn=t2m_contact_collate
            )


    motion_data_loader =torch.utils.data.DataLoader(
        motion_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=False,
        shuffle=False,
        drop_last=True,
        collate_fn=t2hoi_collate
    )
    # for data in motion_data_loader:
    #     print(data)
        # break
    # print(dataset[0])
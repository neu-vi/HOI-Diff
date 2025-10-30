from utils import dist_utils
from data_loaders.behave.utils.word_vectorizer import POS_enumerator
from utils.feature_extractor.modules import TextEncoderBiGRUCo,MotionEncoderBiGRUCo,MovementConvEncoder
import os.path as osp
import torch
import numpy as np
from os.path import join as pjoin

def build_evaluators(opt):
    movement_enc = MovementConvEncoder(opt['dim_pose'], opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])
    text_enc = TextEncoderBiGRUCo(word_size=opt['dim_word'],
                                  pos_size=opt['dim_pos_ohot'],
                                  hidden_size=opt['dim_text_hidden'],
                                  output_size=opt['dim_coemb_hidden'],
                                  device=opt['device'])
    motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_enc_hidden'],
                                      hidden_size=opt['dim_motion_hidden'],
                                      output_size=opt['dim_coemb_hidden'],
                                      device=opt['device'])
    checkpoint=torch.load(osp.join(opt['checkpoint_dir'],'model','finest.tar'),
                          map_location=opt['device'])
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))

    return text_enc,motion_enc,movement_enc

class EvaluationWrapper:
    def __init__(self,val_args):
        opt={
            'device':dist_utils.dev(),
            'dim_word':300,
            'max_motion_length':300,
            'dim_pos_ohot': len(POS_enumerator),
            'dim_motion_hidden': 1024,
            'max_text_len':40,
            'dim_text_hidden':512,
            'dim_coemb_hidden':512,
            'dim_pose':263-4,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoint_dir':'./t2hoi/omomo_fe',
            'unit_length':5
        }
        self.text_encoder,self.motion_encoder,self.movement_encoder=build_evaluators(opt)
        self.opt = opt
        self.device = opt['device']

        self.text_encoder.to(opt['device'])
        self.motion_encoder.to(opt['device'])
        self.movement_encoder.to(opt['device'])

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
    
    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    def get_motion_embeddings(self,motions,m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

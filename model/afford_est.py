import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
# from model.points_encoder import PointNet2Encoder
from .mdm import *


class AffordEstimation(nn.Module):
    def __init__(self,  latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                activation="gelu", clip_dim=512, clip_version=None, **kargs):
        super().__init__()


        self.latent_dim = 256

        self.ff_size = 512
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim



        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)


        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        self.objEmbedding = PointNet2Encoder(c_in=0, c_out=self.latent_dim, num_keypoints=256)
        # self.objEmbedding = nn.Linear(3, self.latent_dim)


        print("TRANS_ENC init")
        seqTransEncoderLayer_contact = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder_contact = nn.TransformerEncoder(seqTransEncoderLayer_contact,
                                                        num_layers=self.num_layers)

        
        seqTransEncoderLayer_pos = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder_pos = nn.TransformerEncoder(seqTransEncoderLayer_pos,
                                                        num_layers=self.num_layers)



        self.output_contact_process = nn.Linear(self.latent_dim, 4)
        self.input_contact_process = nn.Linear(4, self.latent_dim)

            


        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
    
    def mask_cond_obj(self, cond, force_mask=False):
        seq, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def encode_obj(self, obj_points):
        # obj_points - [bs, n_points, 3]
        obj_emb = self.objEmbedding(obj_points) # [bs, n_points, d]
        obj_emb = obj_emb.permute(1, 0, 2) # [n_points, bs, d]
        return obj_emb
    
    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        bs, njoints, nfeats, nframes = x.shape



        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        # encode object shape
        if 'obj_points' in y.keys():
            enc_obj = self.encode_obj(y['obj_points'])
            emb = emb + self.mask_cond_obj(enc_obj, force_mask=force_mask)




        x = x.permute(3,0,1,2).reshape(nframes, bs, njoints*nfeats)
        contact_input = self.input_contact_process(x)

        xseq_contact = torch.cat((emb, contact_input), axis=0)  # [seqlen+256 , bs, d]
        xseq_contact = self.sequence_pos_encoder(xseq_contact)  # [seqlen+1, bs, d]



        output_contact = self.seqTransEncoder_contact(xseq_contact)[256:]  # [seqlen+256, bs, d]

        output = self.output_contact_process(output_contact)
        output = output.reshape(nframes, bs, njoints, nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


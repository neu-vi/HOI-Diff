import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
# import tensorflow as tf
from collections import OrderedDict
from os.path import join as pjoin
import codecs as cs
from loguru import logger as log
import numpy as np
import time
import os
from tqdm import tqdm
from utils.feature_extractor.modules import ContrastiveLoss

class TextMotionMatchTrainer(object):

    def __init__(self, args, text_encoder, motion_encoder,movement_encoder, train_platform):
        self.opt = args
        self.text_encoder = text_encoder
        self.motion_encoder = motion_encoder
        self.movement_encoder=movement_encoder
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            self.logger = train_platform
            self.contrastive_loss = ContrastiveLoss(self.opt.negative_margin)

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.text_encoder.load_state_dict(checkpoints['text_encoder'])
        self.motion_encoder.load_state_dict(checkpoints['motion_encoder'])
        self.movement_encoder.load_state_dict(checkpoints['movement_encoder'])

        self.opt_text_encoder.load_state_dict(checkpoints['opt_text_encoder'])
        self.opt_motion_encoder.load_state_dict(checkpoints['opt_motion_encoder'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'text_encoder': self.text_encoder.state_dict(),
            'motion_encoder': self.motion_encoder.state_dict(),
            'movement_encoder': self.movement_encoder.state_dict(),
            'opt_text_encoder': self.opt_text_encoder.state_dict(),
            'opt_motion_encoder': self.opt_motion_encoder.state_dict(),
            'epoch': epoch,
            'iter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def to(self, device):
        self.text_encoder.to(device)
        self.motion_encoder.to(device)
        self.movement_encoder.to(device)

    def train_mode(self):
        self.text_encoder.train()
        self.motion_encoder.train()
        self.movement_encoder.eval()

    def forward(self, batch_data):
        word_emb, pos_ohot, caption, cap_lens, motions, m_lens, _ = batch_data
        word_emb = word_emb.detach().to(self.device).float()
        pos_ohot = pos_ohot.detach().to(self.device).float()
        motions = motions.detach().to(self.device).float()

        # Sort the length of motions in descending order, (length of text has been sorted)
        self.align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        # log.info(self.align_idx)
        # log.info(m_lens[self.align_idx])
        motions = motions[self.align_idx]
        m_lens = m_lens[self.align_idx]

        '''Movement Encoding'''
        movements = self.movement_encoder(motions).detach()
        m_lens=m_lens//self.opt.unit_length
        self.motion_embedding = self.motion_encoder(movements, m_lens)

        '''Text Encoding'''
        # time0 = time.time()
        # text_input = torch.cat([word_emb, pos_ohot], dim=-1)
        self.text_embedding = self.text_encoder(word_emb, pos_ohot, cap_lens)
        self.text_embedding = self.text_embedding.clone()[self.align_idx]


    def backward(self):

        batch_size = self.text_embedding.shape[0]
        '''Positive pairs'''
        pos_labels = torch.zeros(batch_size).to(self.text_embedding.device)
        self.loss_pos = self.contrastive_loss(self.text_embedding, self.motion_embedding, pos_labels)

        '''Negative Pairs, shifting index'''
        neg_labels = torch.ones(batch_size).to(self.text_embedding.device)
        shift = np.random.randint(0, batch_size-1)
        new_idx = np.arange(shift, batch_size + shift) % batch_size
        self.mis_motion_embedding = self.motion_embedding.clone()[new_idx]
        self.loss_neg = self.contrastive_loss(self.text_embedding, self.mis_motion_embedding, neg_labels)
        self.loss = self.loss_pos + self.loss_neg

        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss.item()
        loss_logs['loss_pos'] = self.loss_pos.item()
        loss_logs['loss_neg'] = self.loss_neg.item()
        return loss_logs


    def update(self):

        self.zero_grad([self.opt_motion_encoder, self.opt_text_encoder])
        loss_logs = self.backward()
        self.loss.backward()
        self.clip_norm([self.text_encoder, self.motion_encoder])
        self.step([self.opt_text_encoder, self.opt_motion_encoder])

        return loss_logs


    def train(self, train_dataloader, val_dataloader):
        self.to(self.device)

        self.opt_motion_encoder = optim.Adam(self.motion_encoder.parameters(), lr=self.opt.lr)
        self.opt_text_encoder = optim.Adam(self.text_encoder.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        log.info('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        logs = OrderedDict()

        min_val_loss = np.inf
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            log.info('Epoch: %03d/ %d' % (epoch, self.opt.max_epoch))
            for i, batch_data in enumerate(train_dataloader):
                self.train_mode()

                self.forward(batch_data)
                # time3 = time.time()
                log_dict = self.update()
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v


                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    self.logger.report_scalar('val_loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.report_scalar(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    # print_current_loss_decomp(start_time, it, total_iters, mean_loss, epoch, i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, it)

            log.info('Validation time:')

            loss_pos_pair = 0
            loss_neg_pair = 0
            val_loss = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    loss_pos_pair += self.loss_pos.item()
                    loss_neg_pair += self.loss_neg.item()
                    val_loss += self.loss.item()

            loss_pos_pair /= len(val_dataloader) + 1
            loss_neg_pair /= len(val_dataloader) + 1
            val_loss /= len(val_dataloader) + 1
            log.info('Validation Loss: %.5f Positive Loss: %.5f Negative Loss: %.5f' %
                  (val_loss, loss_pos_pair, loss_neg_pair))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss

            if epoch % self.opt.eval_every_e == 0:
                pos_dist = F.pairwise_distance(self.text_embedding, self.motion_embedding)
                neg_dist = F.pairwise_distance(self.text_embedding, self.mis_motion_embedding)

                pos_str = ' '.join(['%.3f' % (pos_dist[i]) for i in range(pos_dist.shape[0])])
                neg_str = ' '.join(['%.3f' % (neg_dist[i]) for i in range(neg_dist.shape[0])])

                save_path = pjoin(self.opt.eval_dir, 'E%03d.txt' % (epoch))
                with cs.open(save_path, 'w') as f:
                    f.write('Positive Pairs Distance\n')
                    f.write(pos_str + '\n')
                    f.write('Negative Pairs Distance\n')
                    f.write(neg_str + '\n')

class DecompTrainerV3(object):
    def __init__(self, args, movement_enc, movement_dec,train_platform):
        self.opt = args
        self.movement_enc = movement_enc
        self.movement_dec = movement_dec
        self.device = args.device

        if args.is_train:
            self.logger = train_platform
            self.sml1_criterion = torch.nn.SmoothL1Loss()
            self.l1_criterion = torch.nn.L1Loss()
            self.mse_criterion = torch.nn.MSELoss()


    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data):
        motions = batch_data
        self.motions = motions.detach().to(self.device).float()
        self.latents = self.movement_enc(self.motions)
        self.recon_motions = self.movement_dec(self.latents)

    def backward(self):
        self.loss_rec = self.l1_criterion(self.recon_motions, self.motions)
                        # self.sml1_criterion(self.recon_motions[:, 1:] - self.recon_motions[:, :-1],
                        #                     self.motions[:, 1:] - self.recon_motions[:, :-1])
        self.loss_sparsity = torch.mean(torch.abs(self.latents))
        self.loss_smooth = self.l1_criterion(self.latents[:, 1:], self.latents[:, :-1])
        self.loss = self.loss_rec + self.loss_sparsity * self.opt.lambda_sparsity +\
                    self.loss_smooth*self.opt.lambda_smooth

    def update(self):
        # time0 = time.time()
        self.zero_grad([self.opt_movement_enc, self.opt_movement_dec])
        # time1 = time.time()
        # print('\t Zero_grad Time: %.5f s' % (time1 - time0))
        self.backward()
        # time2 = time.time()
        # print('\t Backward Time: %.5f s' % (time2 - time1))
        self.loss.backward()
        # time3 = time.time()
        # print('\t Loss backward Time: %.5f s' % (time3 - time2))
        # self.clip_norm([self.movement_enc, self.movement_dec])
        # time4 = time.time()
        # print('\t Clip_norm Time: %.5f s' % (time4 - time3))
        self.step([self.opt_movement_enc, self.opt_movement_dec])
        # time5 = time.time()
        # print('\t Step Time: %.5f s' % (time5 - time4))

        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss_rec.item()
        loss_logs['loss_rec'] = self.loss_rec.item()
        loss_logs['loss_sparsity'] = self.loss_sparsity.item()
        loss_logs['loss_smooth'] = self.loss_smooth.item()
        return loss_logs

    def save(self, file_name, ep, total_it):
        state = {
            'movement_enc': self.movement_enc.state_dict(),
            'movement_dec': self.movement_dec.state_dict(),

            'opt_movement_enc': self.opt_movement_enc.state_dict(),
            'opt_movement_dec': self.opt_movement_dec.state_dict(),

            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)
        return

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)

        self.movement_dec.load_state_dict(checkpoint['movement_dec'])
        self.movement_enc.load_state_dict(checkpoint['movement_enc'])

        self.opt_movement_enc.load_state_dict(checkpoint['opt_movement_enc'])
        self.opt_movement_dec.load_state_dict(checkpoint['opt_movement_dec'])

        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_dataloader, val_dataloader):
        self.movement_enc.to(self.device)
        self.movement_dec.to(self.device)

        self.opt_movement_enc = optim.Adam(self.movement_enc.parameters(), lr=self.opt.lr)
        self.opt_movement_dec = optim.Adam(self.movement_dec.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        log.info('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        logs = OrderedDict()
        while epoch < self.opt.max_epoch:
            log.info('Epoch: %03d/ %d' % (epoch, self.opt.max_epoch))
            # time0 = time.time()
            for i, batch_data in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
                self.movement_dec.train()
                self.movement_enc.train()

                # time1 = time.time()
                # print('DataLoader Time: %.5f s'%(time1-time0) )
                self.forward(batch_data)
                # time2 = time.time()
                # print('Forward Time: %.5f s'%(time2-time1))
                log_dict = self.update()
                # time3 = time.time()
                # print('Update Time: %.5f s' % (time3 - time2))
                # time0 = time3
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    self.logger.report_scalar('val_loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.report_scalar(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    # print_current_loss_decomp(start_time, it, total_iters, mean_loss, epoch, i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            log.info('Validation time:')

            val_loss = 0
            val_rec_loss = 0
            val_sparcity_loss = 0
            val_smooth_loss = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    val_rec_loss += self.loss_rec.item()
                    val_smooth_loss += self.loss.item()
                    val_sparcity_loss += self.loss_sparsity.item()
                    val_smooth_loss += self.loss_smooth.item()
                    val_loss += self.loss.item()

            val_loss = val_loss / (len(val_dataloader) + 1)
            val_rec_loss = val_rec_loss / (len(val_dataloader) + 1)
            val_sparcity_loss = val_sparcity_loss / (len(val_dataloader) + 1)
            val_smooth_loss = val_smooth_loss / (len(val_dataloader) + 1)
            log.info('Validation Loss: %.5f Reconstruction Loss: %.5f '
                  'Sparsity Loss: %.5f Smooth Loss: %.5f' % (val_loss, val_rec_loss, val_sparcity_loss, \
                                                             val_smooth_loss))

            if epoch % self.opt.eval_every_e == 0:
                data = torch.cat([self.recon_motions, self.motions], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                
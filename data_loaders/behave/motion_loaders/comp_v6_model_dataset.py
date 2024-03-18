import os
import copy
import torch
from os.path import join as pjoin
from tqdm import tqdm
from utils import dist_util
from data_loaders.behave.networks.modules import *
from data_loaders.behave.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from data_loaders.behave.data.dataset import global3d_to_rel, sample_to_motion
from data_loaders.behave.utils.metrics import calculate_skating_ratio
from sample.condition import Guide_Contact
from utils.fixseed import fixseed
from  data_loaders.behave.scripts.motion_process import *
def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator

class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
        trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6
        # print(mm_idxs)

        print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                          m_lens[0]//opt.unit_length, opt.dim_pose)
                    if t == 0:
                        # print(m_lens)
                        # print(text_data)
                        sub_dict = {'motion': pred_motions[0].cpu().numpy(),
                                    'length': m_lens[0].item(),
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens}
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                if is_mm:
                    mm_generated_motions.append({'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer



    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompMDMGeneratedDataset(Dataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)




class CompMDMGeneratedDatasetCondition(Dataset):

    def __init__(self, model_dict, diffusion_dict, dataloader, mm_num_samples, mm_num_repeats, 
                 max_motion_length, num_samples_limit, scale=1., save_dir=None, afford_save_dir=None, skip_first_stage=True, 
                 wo_obj_motion=False,
                 seed=None):
        
        assert seed is not None, "must provide seed"

        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.save_dir = save_dir
        self.afford_save_dir = afford_save_dir
        self.sigmoid = torch.nn.Sigmoid()

        self.skip_first_stage = skip_first_stage
        self.wo_obj_motion = wo_obj_motion


        motion_model, afford_model = model_dict["motion"], model_dict["afford"]
        motion_diffusion, afford_diffusion = diffusion_dict["motion"], diffusion_dict["afford"]

        ### Basic settings
        motion_classifier_scale = 100.0
        print("motion classifier scale", motion_classifier_scale)
        log_motion = False
        guidance_mode = 'no'
        print("guidance mode", guidance_mode)

        model_device = next(motion_model.parameters()).device


        # motion_diffusion.data_get_mean_fn = self.dataset.t2m_dataset.get_std_mean
        # motion_diffusion.data_transform_fn = self.dataset.t2m_dataset.transform_th
        # motion_diffusion.data_inv_transform_fn = self.dataset.t2m_dataset.inv_transform_th


        if afford_diffusion is not None:        
            afford_sample_fn = afford_diffusion.p_sample_loop
            afford_model.eval()
        else:
            # If we don't have a trajectory diffusion model, assume that we are using classifier-free 1-stage model
            pass

        assert save_dir is not None
        assert afford_save_dir is not None
        assert mm_num_samples < len(dataloader.dataset)

        # create the target directory
        os.makedirs(self.save_dir, exist_ok=True)

        # use_ddim = False  # FIXME - hardcoded
        # NOTE: I have updated the code in gaussian_diffusion.py so that it won't clip denoise for xstart models.
        # hence, always set the clip_denoised to True
        clip_denoised = True
        self.max_motion_length = max_motion_length

        sample_motion_fn = motion_diffusion.p_sample_loop


        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        # NOTE: mm = multi-modal
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        motion_model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):
                '''For each datapoint, we do the following
                    1. Sample affordance information
                    2. Generate human object inteaction
                    3. using affordance information for guidance if allowed
                '''

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]
                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale


                # input to cuda
                model_kwargs['y']['obj_points'] = model_kwargs['y']['obj_points'].to(dist_util.dev())
                model_kwargs['y']['obj_normals'] = model_kwargs['y']['obj_normals'].to(dist_util.dev())
                
                
                # model_kwargs['y']['gt_afford_data'] = model_kwargs['y']['gt_afford_data'].to(dist_util.dev())
                # Convert to 3D motion space
                # NOTE: the 'motion' will not be random projected if dataset mode is 'eval' or 'gt', 
                # even if the 'self.dataset.t2m_dataset.use_rand_proj' is True

                # # print(f" ============={motion.shape} ")
                # gt_poses = motion.permute(0, 2, 3, 1)
                # # gt_poses = gt_poses * self.dataset.std + self.dataset.mean  # [bs, 1, 196, 263]
                # # (x,y,z) [bs, 1, 120, njoints=22, nfeat=3]
                # gt_skel_motions = gt_poses[...,:66].reshape(gt_poses.shape[0], -1, 22 ,3)
                # # gt_skel_motions = gt_skel_motions.view(-1, *gt_skel_motions.shape[2:]).permute(0, 2, 3, 1)
                # # gt_skel_motions = motion_model.rot2xyz(x=gt_skel_motions, mask=None, pose_rep='xyz', glob=True, translation=True, 
                # #                                     jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False)
                # # gt_skel_motions shape [32, 22, 3, 196]
                # # # Visualize to make sure it is correct
                # from data_loaders.humanml.utils.plot_script import plot_3d_motion
                # from data_loaders.behave.utils.paramUtil import t2m_kinematic_chain
                # plot_3d_motion("./test_positions_1.mp4", t2m_kinematic_chain, 
                #                gt_skel_motions[0].detach().cpu().numpy(), 'title', 'humanml', fps=20)
                
                # Next, sample points, then prepare target and inpainting mask for trajectory model
                ## Sample points
                n_keyframe = 5
                # reusing the target if it exists
                target_batch_file = f'target_{i:04d}.pt'
                target_batch_file = os.path.join(self.save_dir, target_batch_file)
                if os.path.exists(target_batch_file):
                    # [batch_size, n_keyframe]
                    sampled_keyframes = torch.load(target_batch_file, map_location=motion.device)
                    print(f'sample keyframes {target_batch_file} exists, loading from file')
                else:
                    sampled_keyframes = torch.rand(motion.shape[0], n_keyframe) * model_kwargs['y']['lengths'].unsqueeze(-1)
                    # Floor to int because ceil to 'lengths' will make the idx out-of-bound.
                    # The keyframe can be a duplicate.
                    sampled_keyframes = torch.floor(sampled_keyframes).int().sort()[0]  # shape [batch_size, n_keyframe]
                    torch.save(sampled_keyframes, target_batch_file)
                # import pdb; pdb.set_trace()

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                mm_trajectories = []
                for t in range(repeat_times):
                    seed_number = seed * 100_000 + i * 100 + t
                    fixseed(seed_number)
                    batch_file = f'{i:04d}_{t:02d}.pt'
                    afford_batch_file = f'af_{i:04d}_{t:02d}.pt'
                    batch_path = os.path.join(self.save_dir, batch_file)
                    afford_batch_path = os.path.join(self.afford_save_dir, afford_batch_file)
                    # afford_batch_path = os.path.join(self.save_dir, afford_batch_file)
                    if os.path.exists(afford_batch_path):
                        # [bs, njoints, nfeat, seqlen]
                        afford_sample = torch.load(afford_batch_path, map_location=motion.device)
                        print(f'afford batch {afford_batch_file} exists, loading from file')

                    else: 
                        print(f'working on {afford_batch_file}')

                        ### 1. Prepare motion for conditioning ###
                        afford_model_kwargs = copy.deepcopy(model_kwargs)

                        if skip_first_stage:
                            skip_timesteps = 450
                        else:
                            skip_timesteps = 0
                        ### Generate trajectory
                        afford_sample = afford_sample_fn(
                                        afford_model,
                                        (motion.shape[0], 4, afford_model.nfeats, 8),    # + 6 object pose
                                        clip_denoised=False,
                                        model_kwargs=afford_model_kwargs,
                                        skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
                                        init_image=None,
                                        progress=True,
                                        dump_steps=None,
                                        noise=None,
                                        const_noise=False,
                                        cond_fn=None,
                                    )
                        ### Generate trajectory
                        # [bs, njoints, nfeat, seqlen]
                        torch.save(afford_sample, afford_batch_path)
                    
                                        # Post-processing estimated affordance data
                    sigmoid = torch.nn.Sigmoid()

                    afford_sample[:,3:] = sigmoid(afford_sample[:,3:])

                    # afford_sample = afford_sample[...,0].squeeze().to(dist_util.dev())
                    # sample_contact = afford_sample[...,:6].reshape(-1, 2, 3)
                    # h_afford_labels = torch.zeros([afford_sample.shape[0], 22]).to(dist_util.dev())
                    # temp = sigmoid(afford_sample[...,6:14])
                    # # skel_labels = (sigmoid(afford_sample[...,6:14])>0.6).int()
                    # # is_static = (sigmoid(afford_sample[...,14:])>0.6).int()

                    # skel_labels = sigmoid(afford_sample[...,6:14])
                    # is_static = sigmoid(afford_sample[...,14:])

                    # sel_joints = [0,9,10,11,16,17,20,21]
                    # for i in range(afford_sample.shape[0]):
                    #     h_afford_labels[i, sel_joints] = temp[i]
                    # o_afford_labels = torch.zeros(0).to(dist_util.dev())
                    # all_contact_points = torch.zeros(0).to(dist_util.dev())
                    # h_mask = (h_afford_labels>0.5).int()
                    # labels, ind = torch.topk((h_afford_labels * h_mask), 2)

                    # for i in range(afford_sample.shape[0]):
                    #     points = model_kwargs['y']['obj_points'][i]
                    #     normals = model_kwargs['y']['obj_normals'][i]
                    #     dist = torch.cdist(sample_contact[i], points)
                    #     min_dist_idx = torch.argmin(dist, dim=-1)
                    #     o_afford_labels = torch.cat([o_afford_labels, min_dist_idx.unsqueeze(0)])

                    #     if labels[i, 1] < 0.7:
                    #         ind[i, 1:2] = ind[i, 0:1]
                    #     if labels[i, 0] < 0.7:
                    #         ind[i, 0:1] = ind[i, 1:2]                            

                    #     # Choose a specific point on the object's surface (e.g., vertex 0)
                    #     chosen_point1 = points[min_dist_idx[0]]
                    #     chosen_point2 = points[min_dist_idx[1]]
                    #     normal_at_chosen_point1 = normals[min_dist_idx[0]]
                    #     normal_at_chosen_point2 = normals[min_dist_idx[1]]
                        
                    #     distance = 0.05  # Adjust as needed

                    #     # Calculate the direction vector away from the chosen point
                    #     direction_vector1 = normal_at_chosen_point1 * distance
                    #     direction_vector2 = normal_at_chosen_point2 * distance
                        

                    #     # Obtain the position outside the chosen point on the surface
                    #     position_outside1 = chosen_point1 + direction_vector1
                    #     position_outside2 = chosen_point2 + direction_vector2

                    #     contact_points = torch.cat([chosen_point1, chosen_point2], 0) 
                    #     # contact_points = torch.cat([position_outside1, position_outside2], 0)  
                    #     all_contact_points = torch.cat([all_contact_points, contact_points.unsqueeze(0)], dim=0)
                    #         # afford_labels = torch.cat([h_afford_labels, o_afford_labels], dim=-1)

                    #         # afford_labels = torch.cat([h_afford_labels, all_contact_points], dim=-1)

                    #     afford_labels = torch.cat([ind, is_static], -1)

                    # model_kwargs['y']['obj_points'] = torch.cat([model_kwargs['y']['obj_points'], all_contact_points.reshape(-1, 2, 3)], 1)
                    # model_kwargs['y']['afford_labels'] = afford_labels.to(dist_util.dev())

                    # h_afford_data = temp
                    # pred_afford_data = torch.cat([all_contact_points, skel_labels, is_static], -1)


                    # reusing the batch if it exists
                    if os.path.exists(batch_path):
                        # [bs, njoints, nfeat, seqlen]
                        sample_motion = torch.load(batch_path, map_location=motion.device)
                        print(f'batch {batch_file} exists, loading from file')
                    else:                        
                        print(f'working on {batch_file}')
       
                        # skip_first_stage_ = False
                        if skip_first_stage:
                            # No first stage. Skip straight to second stage 
                            guide_fn_contact = None
                            pass

                        else:
                            guide_fn_contact = Guide_Contact(
                                                inv_transform_th=self.dataset.t2m_dataset.inv_transform_th,
                                                mean=self.dataset.t2m_dataset.mean,
                                                std=self.dataset.t2m_dataset.std,
                                                classifiler_scale=motion_classifier_scale,
                                                use_global=self.dataset.use_global,
                                                batch_size=afford_sample.shape[0],
                                                afford_sample = afford_sample
                                                )
    

                        sample_motion = sample_motion_fn(
                            motion_model,
                            motion.shape,    # + 6 object pose
                            clip_denoised=False,
                            model_kwargs=model_kwargs,
                            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                            init_image=None,
                            progress=True,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,
                            cond_fn=guide_fn_contact,
                        )

                        # save to file
                        torch.save(sample_motion, batch_path)

                    # print('cut the motion length from {} to {}'.format(sample_motion.shape[-1], self.max_motion_length))
                    sample = sample_motion[:, :, :, :self.max_motion_length]

    
                    # kps_error = compute_kps_error(cur_motion, gt_skel_motions, sampled_keyframes)  # [batch_size, 5] in meter
                    # skate_ratio, skate_vel = calculate_skating_ratio(cur_motion)  # [batch_size]


                    # NOTE: To test if the motion is reasonable or not
                    log_motion = False
                    if self.dataset.use_global:
                        # cur_motion = sample_to_motion(sample, self.dataset, motion_model)
                        # NOTE: Changing the output from absolute space to the relative space here.
                        # The easiest way to do this is to go all the way to skeleton and convert back again.
                        # sample shape [32, 262, 1, 196]  -> [32, 263, 1, 196]
                        sample, sample_obj = global3d_to_rel(sample, self.dataset, motion_model, is_norm=True)
                        B,_,_,T = sample.shape
                        sample = torch.cat([sample, sample_obj], dim=1)

                        if log_motion:
                            sample_after = (sample.permute(0, 2, 3, 1) *  self.dataset.std_rel) +  self.dataset.mean_rel
                            n_joints =22
                            sample_global = recover_from_ric(sample_after.float(), n_joints)
                            sample_global = sample_global[:,0,:,:n_joints*3]
                            cur_motion = sample_global.reshape(sample_global.shape[0], sample_global.shape[1], 22, 3).permute(0,2,3,1)
                    
                            from data_loaders.humanml.utils.plot_script import plot_3d_motion
                            for j in tqdm([1, 3, 4, 5], desc="generating motion"):
                                motion_id = f'{i:04d}_{t:02d}_{j:02d}'
                                plot_3d_motion(os.path.join(self.save_dir, f"motion_cond_{motion_id}.mp4"), self.dataset.kinematic_chain, 
                                cur_motion[j].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)
                    else:
         
                        if log_motion:
                            sample_copy = sample.clone()
                            sample_copy = self.dataset.t2m_dataset.inv_transform(sample_copy.cpu().permute(0, 2, 3, 1)).float()

                            sample_obj = sample_copy[..., 263:]
                            sample_obj = sample_obj.permute(0, 1, 3, 2)
                            sample_copy = sample_copy[..., :263]
                            n_joints = 22

                            sample_global = recover_from_ric(sample_copy, n_joints)
                            sample_global = sample_global[:,0,:,:n_joints*3]
                            cur_motion = sample_global.reshape(sample_global.shape[0], sample_global.shape[1], 22, 3).permute(0,2,3,1)

                            from data_loaders.humanml.utils.plot_script import plot_3d_motion
                            for j in tqdm([1, 3, 4, 5], desc="generating motion"):
                                motion_id = f'{i:04d}_{t:02d}_{j:02d}'
                                plot_3d_motion(os.path.join(self.save_dir, f"motion_cond_{motion_id}.mp4"), self.dataset.kinematic_chain, 
                                cur_motion[j].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)


                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'obj_points': model_kwargs['y']['obj_points'][bs_i].cpu().numpy(),
                                    'gt_afford_data': model_kwargs['y']['gt_afford_data'][bs_i].cpu().numpy(),
                                    'pred_afford_data':afford_sample[bs_i].cpu().numpy(),
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        # 'traj': cur_traj[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]
                        # import pdb; pdb.set_trace()

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, obj_points, gt_afford_data, pred_afford_data = data['motion'], data['length'], data['caption'], data['tokens'], data['obj_points'], data['gt_afford_data'], data['pred_afford_data']
        # rel_distance_error = data['rel_distance_error']
        # skate_ratio = data['skate_ratio']
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            if self.dataset.use_global:
                # Denorm with rel_transform because the inv_transform() will have the absolute mean and std
                # The motion is already converted to relative after inference
                # import pdb; pdb.set_trace()
                denormed_motion = (normed_motion * self.dataset.std_rel) + self.dataset.mean_rel
            else:    
                # print(f"======= {normed_motion.shape}")
                denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)

            
            renormed_motion = (denormed_motion[...,:263] - self.dataset.mean_for_eval[:263]) / self.dataset.std_for_eval[:263]  # according to T2M norms
            # renormed_motion = (denormed_motion[...,:263] - self.dataset.t2m_dataset.mean[:263]) / self.dataset.t2m_dataset.std[:263]  # according to T2M norms
            # renormed_motion = denormed_motion[...,:263]
            
            motion = renormed_motion

            if not self.wo_obj_motion:
                motion_obj = denormed_motion[..., 263:]

            # This step is needed because T2M evaluato rs expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)


        if not self.wo_obj_motion:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, motion_obj,  m_length, '_'.join(tokens), obj_points, gt_afford_data, pred_afford_data
        else:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), obj_points, gt_afford_data, pred_afford_data
    
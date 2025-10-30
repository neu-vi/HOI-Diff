# import os
# import os.path as osp

# from loguru import logger
# import torch
# from utils.parser_util import train_feature_extractor_args
# from utils.misc import fixseed
# from data_loaders.behave.utils.word_vectorizer import POS_enumerator,WordVectorizer
# from utils.feature_extractor.modules import TextEncoderBiGRUCo,MotionEncoderBiGRUCo,MovementConvEncoder
# from utils.feature_extractor.trainer import TextMotionMatchTrainer
# from dataset.fe_dataset import feature_extractor_dataset, collate_fn
from train.train_platforms import WandbPlatform

# from torch.utils.data import DataLoader

# def build_models(args):
#     movement_enc=MovementConvEncoder(dim_pose,args.dim_movement_enc_hidden,args.dim_movement_latent)
#     text_enc = TextEncoderBiGRUCo(word_size=dim_word,
#                                   pos_size=dim_pos_ohot,
#                                   hidden_size=args.dim_text_hidden,
#                                   output_size=args.dim_coemb_hidden,
#                                   device=args.device)
#     motion_enc = MotionEncoderBiGRUCo(input_size=args.dim_movement_latent,
#                                       hidden_size=args.dim_motion_hidden,
#                                       output_size=args.dim_coemb_hidden,
#                                       device=args.device)
    
#     if not args.is_continue:
#        logger.info('Loading Decomp......')
#        checkpoint = torch.load(osp.join(args.checkpoints_dir, args.decomp_name, 'model', 'latest.tar'),
#                                map_location=args.device)
#        movement_enc.load_state_dict(checkpoint['movement_enc'])
#     return text_enc,motion_enc,movement_enc

# if __name__=='__main__':
#     args=train_feature_extractor_args()
#     args.device = torch.device("cpu" if args.gpu_id==-1 else "cuda:" + str(args.gpu_id))
#     torch.autograd.set_detect_anomaly(True)
#     fixseed(args.seed)
#     if args.gpu_id!=-1:
#         torch.cuda.set_device(args.gpu_id)
#     args.save_path=osp.join(args.save_dir,args.exp_name)
#     args.model_dir=osp.join(args.save_path,'model')
#     args.log_dir=osp.join(args.save_path,'log')
#     args.eval_dir=osp.join(args.save_path,'eval')

#     os.makedirs(args.save_path,exist_ok=True)
#     os.makedirs(args.model_dir,exist_ok=True)
#     os.makedirs(args.log_dir,exist_ok=True)
#     os.makedirs(args.eval_dir,exist_ok=True)

#     logger.add(osp.join(args.log_dir,'train_feature_extractor.log'),rotation='10 MB')

#     args.data_root='./dataset/omomo_t2m'
#     args.max_motion_length=50
#     dim_pose=259

#     meta_root=osp.join('./glove')
#     dim_word=300
#     dim_pos_ohot=len(POS_enumerator)

#     w_vectorizer=WordVectorizer(meta_root,'our_vab')
#     text_encoder,motion_encoder,movement_encoder=build_models(args)

#     pc_text_enc = sum(param.numel() for param in text_encoder.parameters())
#     logger.info("Total parameters of text encoder: {}".format(pc_text_enc))
#     pc_motion_enc = sum(param.numel() for param in motion_encoder.parameters())
#     logger.info("Total parameters of motion encoder: {}".format(pc_motion_enc))
#     logger.info("Total parameters: {}".format(pc_motion_enc + pc_text_enc))

#     train_platform=WandbPlatform(args.save_path)

#     trainer=TextMotionMatchTrainer(args,text_encoder,motion_encoder,movement_encoder,train_platform)

#     train_dataset=feature_extractor_dataset(args,'train',w_vectorizer)
#     val_dataset=feature_extractor_dataset(args,'test',w_vectorizer)

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=4,
#                               shuffle=True, collate_fn=collate_fn, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, num_workers=4,
#                             shuffle=True, collate_fn=collate_fn, pin_memory=True)
    
#     trainer.train(train_loader,val_loader)



import os

from os.path import join as pjoin
import torch


from data_loaders.get_data import get_dataset_loader
from data_loaders.behave.options.train_options import TrainTexMotMatchOptions

from data_loaders.behave.networks.modules import *
from data_loaders.behave.networks.trainers import TextMotionMatchTrainer
from data_loaders.behave.data.dataset import Text2MotionDatasetV2, collate_fn
from data_loaders.behave.scripts.motion_process import *
from torch.utils.data import DataLoader
from data_loaders.behave.utils.word_vectorizer import WordVectorizer, POS_enumerator
from copy import deepcopy



def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose - opt.foot_contact_entries, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=dim_word,
                                  pos_size=dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)
    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    if not opt.is_continue:
       checkpoint = torch.load(pjoin('/work/vig/xiaogangp/repos/Human_Motion/motion-diffusion-model/t2m', opt.decomp_name, 'model', 'latest.tar'),
                               map_location=opt.device)
       movement_enc.load_state_dict(checkpoint['movement_enc'])
    return text_enc, motion_enc, movement_enc


if __name__ == '__main__':
    parser = TrainTexMotMatchOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)
    opt.eval_dir = pjoin(opt.save_root, 'eval')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    opt.foot_contact_entries = 4
    opt.data_root = './dataset/behave_t2m'
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs_local')
    # opt.afford_dir = '/work/vig/xiaogangp/projects/HOI-Diff/dataset/behave_t2m/affordance_data'
    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.joints_num = 22
    opt.max_motion_length = 196
    opt.dim_pose = 263
    num_classes = 200 // opt.unit_length
    meta_root = opt.data_root

    opt.dataset = opt.dataset_name  # For clearml
    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)


    text_encoder, motion_encoder, movement_encoder = build_models(opt)

    pc_text_enc = sum(param.numel() for param in text_encoder.parameters())
    print(text_encoder)
    print("Total parameters of text encoder: {}M".format(pc_text_enc//1e6))
    pc_motion_enc = sum(param.numel() for param in motion_encoder.parameters())
    print(motion_encoder)
    print("Total parameters of motion encoder: {}M".format(pc_motion_enc//1e6))
    print("Total parameters: {}M".format((pc_motion_enc + pc_text_enc)//1e6))

    # train_platform=WandbPlatform(args.checkpoints_dir)
    trainer = TextMotionMatchTrainer(opt, text_encoder, motion_encoder, movement_encoder)

    if opt.dataset_name == 'babel':
        train_loader = get_dataset_loader('babel',
                                          batch_size=opt.batch_size, num_frames=480,  # not in use
                                          split='train', load_mode='evaluator_train', opt=opt)
        val_loader = get_dataset_loader('babel',
                                        batch_size=opt.batch_size, num_frames=480,  # not in use
                                        split='val', load_mode='evaluator_train', opt=opt)

    else:
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        # meta_root = '/work/vig/xiaogangp/projects/HOI-Diff/dataset/behave_t2m'
        mean = np.load(pjoin(meta_root, 'Mean_local.npy'))
        std = np.load(pjoin(meta_root, 'Std_local.npy'))
        # mean = np.load(pjoin(meta_root, 'Mean_local.npy'))
        # std = np.load(pjoin(meta_root, 'std.npy'))
        dataset_args = {
            'opt': opt, 'mean': mean, 'std': std, 'w_vectorizer': w_vectorizer
        }
        train_split_file = pjoin(opt.data_root, 'train.txt')
        val_split_file = pjoin(opt.data_root, 'test.txt')
        val_args, train_args = deepcopy(dataset_args), deepcopy(dataset_args)
        train_args.update({'split_file': train_split_file})
        val_args.update({'split_file': val_split_file})

        train_dataset = Text2MotionDatasetV2(**train_args)
        val_dataset = Text2MotionDatasetV2(**val_args)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                                  shuffle=True, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                                shuffle=True, collate_fn=collate_fn, pin_memory=True)  # FIXME
        # val_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
        #                         shuffle=True, collate_fn=collate_fn, pin_memory=True)

    trainer.train(train_loader, val_loader)
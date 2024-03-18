from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.behave.motion_loaders.model_motion_loaders import get_mdm_loader, get_mdm_loader_cond  # get_motion_loader
from data_loaders.behave.utils.metrics import *
from data_loaders.behave.networks.evaluator_wrapper import EvaluatorMDMWrapper
from collections import OrderedDict
from data_loaders.behave.scripts.motion_process import *
from data_loaders.behave.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model_wo_clip, load_model
from utils.rotation_conversions import axis_angle_to_matrix
from scipy.spatial.transform import Rotation
from diffusion import logger
from utils import dist_util
from data_loaders.behave.data.dataset import global3d_to_rel, sample_to_motion
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
from model.hoi_diff import HOIDiff as used_model
from model.afford_est import AffordEstimation
from diffusion.gaussian_diffusion import LocalMotionDiffusion, AffordDiffusion

torch.multiprocessing.set_sharing_strategy('file_system')

def evaluate_matching_score(eval_wrapper, motion_loaders, file, wo_obj_motion):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        print(motion_loader_name)
        data = motion_loaders[motion_loader_name]

        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                if motion_loader_name == "vald":
                    if not wo_obj_motion:
                        word_embeddings, pos_one_hots, _, sent_lens, motions, motions_obj, m_lens, _, _, _ ,_ = batch
                    else:
                        word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _, _, _ = batch

                else:
                    word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _, _, _, _ = batch
                    motions = motions[...,:263]

                
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )

                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _, _, _, _, _ = batch
            # _, _, _, sent_lens, motions, m_lens, _ = batch

            motions = motions[...,:263]

            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict

# python /work/vig/xiaogangp/codes/hoi-motion_pretrained/utils/gpt_action2text.py
# CUDA_LAUNCH_BLOCKING=1 python -m eval.eval_behave --model_path ./save/hoi_finetuned_obj_local/model000020000.pt  --dataset behave



def evaluate_hoi(motion_loaders, file):
    relative_distance_dict = OrderedDict({})
    skating_ratio_dict = OrderedDict({})
    collision_error_dict = OrderedDict({})

    motion_loader_name = 'vald'
    motion_loader = motion_loaders[motion_loader_name]
    use_global = motion_loader.dataset.dataloader.dataset.use_global
    dist_chamfer_contact = ext.chamferDist()
    print('========== Evaluating HOI ==========')
    # all_dist = []
    all_size = 0
    dist_sum = 0
    skate_ratio_sum = 0
    # dist_err = []
    # traj_err_key = traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
    # print(motion_loader_name)
    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):

            word_embeddings, pos_one_hots, _, sent_lens, motions, motions_obj, m_lens, _, obj_points, gt_afford_data, _  = batch

            # sample to motion
            mean_for_eval = motion_loader.dataset.dataloader.dataset.mean_for_eval
            std_for_eval = motion_loader.dataset.dataloader.dataset.std_for_eval


            # sample_obj = sample_copy[..., 263:]
            # sample_obj = sample_obj.permute(0, 1, 3, 2)
            # sample_copy = sample_copy[..., :263]
            # n_joints = 22

            # sample_global = recover_from_ric(sample_copy, n_joints)
            # sample_global = sample_global[:,0,:,:n_joints*3]
            # cur_motion = sample_global.reshape(sample_global.shape[0], sample_global.shape[1], 22, 3).permute(0,2,3,1)

            motions = motions * std_for_eval[:263] + mean_for_eval[:263]

            # process motion
            motions = motions.float()
            motions_obj = motions_obj.float()


            n_joints = 22 if motions.shape[-1] == 263 else 21
            motions = recover_from_ric(motions, n_joints)


            # sample_global = motions[:,0,:,:n_joints*3]
            # print(f"motions : {motions.shape}")
            # cur_motion = motions.permute(0,2,3,1)

            # from data_loaders.humanml.utils.plot_script import plot_3d_motion
            # for j in tqdm([1, 3, 4, 5], desc="generating motion"):
            #     motion_id = f'{j:04d}'
            #     plot_3d_motion(os.path.join('/work/vig/xiaogangp/codes/hoi-motion_pretrained/save/hoi_finetuned_obj_local/', f"motion_cond_{motion_id}.mp4"), motion_loader.dataset.dataloader.dataset.kinematic_chain, 
            #     cur_motion[j].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)



            if n_joints == 21:
                # kit
                motions = motions * 0.001
            
            # foot skating error
            if n_joints == 21:
                skate_ratio, skate_vel = calculate_skating_ratio_kit(motions.permute(0, 2, 3, 1))  # [batch_size]
            else:
                skate_ratio, skate_vel = calculate_skating_ratio(motions.permute(0, 2, 3, 1))  # [batch_size]
            skate_ratio_sum += skate_ratio.sum()

            # # Relative distance
            # for motion, motion_obj, obj_point, afford_labels in zip(motions, motions_obj, obj_points, gt_afford_data):


            #     h_afford_labels = torch.zeros([afford_labels.shape[0], 22])
            #     h_afford_temp = afford_labels[...,6:14]
            #     sel_joints = [0,9,10,11,16,17,20,21]
            #     h_afford_labels[:, sel_joints] = h_afford_temp.float()
            #     h_idx = np.where(h_afford_labels==1.0)
            #     o_contact_points = afford_labels[:, :6].reshape(2,3)

            #     obj_point_ = o_contact_points   #   Only compute chamfer distance between contact points and contat joints
            #     motion_ = motion[:,h_idx[1]]  # indexed contact joints


            #     # transform
            #     angle, trans = motion_obj[:, :3].transpose(1,0), motion_obj[:, 3:].transpose(1,0)
            #     rot = Rotation.from_rotvec(angle.transpose(1, 0)).as_matrix()
            #     rot_points = np.matmul(obj_point_[np.newaxis], rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + trans.transpose(1, 0)[:, np.newaxis]   
            #     contact_dist, _ = dist_chamfer_contact(motion.float().contiguous().to('cuda:0'),
            #                                                     rot_points.float().contiguous().to('cuda:0'))
            #     contact_dist = torch.mean(torch.sqrt(contact_dist + 1e-4) / (torch.sqrt(contact_dist + 1e-4) + 1.0)) 
            #     dist_sum += contact_dist

            all_size += motions.shape[0]

        # chamfer dist
        # dist_mean = dist_sum / all_size 
        # relative_distance_dict[motion_loader_name] = dist_mean.cpu()

        # Skating evaluation
        skating_score = skate_ratio_sum / all_size
        skating_ratio_dict[motion_loader_name] = skating_score

        # ### For collision error ###
        # collision_err = skate_ratio_sum / all_size
        # collision_error_dict[motion_loader_name] = collision_err

    # print(f'---> [{motion_loader_name}] Contact Distance: {dist_mean:.4f}')
    # print(f'---> [{motion_loader_name}] Contact Distance: {dist_mean:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
    print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}', file=file, flush=True)
    # line = f'---> [{motion_loader_name}] Collision Error: '
    # for (k, v) in zip(traj_err_key, traj_err):
    #     line += '(%s): %.4f ' % (k, np.mean(v))
    # print(line)
    # print(line, file=file, flush=True)
    return relative_distance_dict, skating_ratio_dict



def evaluate_affordance(motion_loaders, file):
                    
    from sklearn.metrics import auc, precision_recall_curve, average_precision_score
    import numpy as np
    import torch

    auc_affordance = OrderedDict({})
    ap_affordance = OrderedDict({})
    l2_distance_contact = OrderedDict({})

    motion_loader_name = 'vald'
    motion_loader = motion_loaders[motion_loader_name]
    use_global = motion_loader.dataset.dataloader.dataset.use_global
    dist_chamfer_contact = ext.chamferDist()
    print('========== Evaluating Affordance ==========')
    # all_dist = []
    all_size = 0
    auc_sum = 0
    ap_sum = 0
    dist_sum = 0
    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):

            word_embeddings, pos_one_hots, _, sent_lens, motions, motions_obj, m_lens, _, obj_points, gt_afford_data, pred_afford_data = batch

            gt_afford_data = gt_afford_data.squeeze()
            

            # Relative distance
            for gt_data, pred_data in zip(gt_afford_data, pred_afford_data):
                gt_labels = gt_data[6:].float()
                gt_contact_points = gt_data[:6].float()

                pred_labels = pred_data[6:].float()
                pred_contact_points = pred_data[:6].float()

            
                # pred = list((pred_labels>0.5).int().cpu().detach().numpy())
                pred = list(pred_labels.cpu().detach().numpy())
                
                gt = list(gt_labels.cpu().detach().numpy())
                # print(f"=========== gt ===========  ")
                # print(f"==========={gt} ===========  ")
                # print(f"=========== pred ===========  ")
                # print(f"=========== {pred} ===========  ")
                precision, recall, th = precision_recall_curve(gt, pred, pos_label=1)
                pr_auc = auc(recall, precision)

                ap = average_precision_score(gt, pred, pos_label=1)
                dist = torch.norm((gt_contact_points.reshape(2, 3) - pred_contact_points.reshape(2, 3)), dim=-1).mean()

                ap_sum += ap
                auc_sum += pr_auc
                dist_sum += dist
                all_size += 1

        # chamfer dist
        auc_mean = auc_sum / all_size
        ap_mean = ap_sum / all_size
        dist_mean = dist_sum / all_size 


        auc_affordance[motion_loader_name] = dist_mean.cpu()
        l2_distance_contact[motion_loader_name] = dist_mean.cpu()
        ap_affordance[motion_loader_name] = dist_mean.cpu()


    print(f'---> [{motion_loader_name}] AUC: {auc_mean:.4f}')
    print(f'---> [{motion_loader_name}] AUC: {auc_mean:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] AP: {ap_mean:.4f}')
    print(f'---> [{motion_loader_name}] AP: {ap_mean:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] L2 Distance: {dist_mean:.4f}')
    print(f'---> [{motion_loader_name}] L2 Distance: {dist_mean:.4f}', file=file, flush=True)
    # line = f'---> [{motion_loader_name}] Collision Error: '
    # for (k, v) in zip(traj_err_key, traj_err):
    #     line += '(%s): %.4f ' % (k, np.mean(v))
    # print(line)
    # print(line, file=file, flush=True)
    return auc_affordance, ap_affordance, l2_distance_contact


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch

                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=False, wo_obj_motion=False):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'Contact_Distance': OrderedDict({}),
                                   'Skating Ratio': OrderedDict({}),
                                   'AUC': OrderedDict({}),
                                   'AP': OrderedDict({}),
                                   'L2-Distance': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                # NOTE: set the seed for each motion loader based on the replication number
                motion_loader, mm_motion_loader = motion_loader_getter(seed=replication)
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

                

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f, wo_obj_motion=wo_obj_motion)

            if not args.wo_obj_motion:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                rel_distance_dict, skating_ratio_dict = evaluate_hoi(motion_loaders, f)

                # print(f'Time: {datetime.now()}')
                # print(f'Time: {datetime.now()}', file=f, flush=True)
                # predict_auc, predict_ap, l2_distance = evaluate_affordance(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            if not wo_obj_motion:
                # for key, item in rel_distance_dict.items():
                #     if key not in all_metrics['Contact_Distance']:
                #         all_metrics['Contact_Distance'][key] = [item]
                #     else:
                #         all_metrics['Contact_Distance'][key] += [item]

                for key, item in skating_ratio_dict.items():
                    if key not in all_metrics['Skating Ratio']:
                        all_metrics['Skating Ratio'][key] = [item]
                    else:
                        all_metrics['Skating Ratio'][key] += [item]
                # for key, item in predict_auc.items():
                #     if key not in all_metrics['AUC']:
                #         all_metrics['AUC'][key] = [item]
                #     else:
                #         all_metrics['AUC'][key] += [item]
                # for key, item in predict_ap.items():
                #     if key not in all_metrics['AP']:
                #         all_metrics['AP'][key] = [item]
                #     else:
                #         all_metrics['AP'][key] += [item]
                # for key, item in l2_distance.items():
                #     if key not in all_metrics['L2-Distance']:
                #         all_metrics['L2-Distance'][key] = [item]
                #     else:
                #         all_metrics['L2-Distance'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict




def load_dataset(args, split, hml_mode, use_global=False, wo_obj_motion=False):
    # (name=args.dataset, batch_size=args.batch_size, num_frames=None, split=split, hml_mode='gt'
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        use_global=use_global,
        split=split,
        num_frames=None,
        hml_mode=hml_mode,
        training_stage=2,
        wo_obj_motion=wo_obj_motion
    ) 
    data = get_dataset_loader(conf)
    return data


if __name__ == '__main__':
    # speed up eval
    torch.set_num_threads(1)
    args = evaluation_parser()
    comment = args.comment
    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), '{}_eval_behave_{}_{}'.format(comment, name, niter))
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += f'_{args.eval_mode}'
    log_file += '.log'

    save_dir = os.path.join(os.path.dirname(args.model_path), f'{comment}_eval_{niter}')
    print(f'Will save to log file [{log_file}]')

    afford_save_dir = os.path.join(os.path.dirname(args.model_path), f'{comment}_eval_{niter}')

    print(f'Eval mode [{args.eval_mode}]')
    if args.eval_mode == 'debug':
        num_samples_limit = 1000  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 200
        replication_times = 5  # about 3 Hrs
    elif args.eval_mode == 'wo_mm':
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 200
        replication_times = 20 # about 12 Hrs
    elif args.eval_mode == 'mm_short':
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 200
        replication_times = 5  # about 15 Hrs
    else:
        raise ValueError()


    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = 'test'

    if args.global_3d:
        print(f"============= Loading Global Pose Rep for Generation")
    else:
        print(f"============= Loading Local Pose Rep for Generation")

    gt_loader = load_dataset(args, split, hml_mode='gt', use_global=False, wo_obj_motion=args.wo_obj_motion)

    gen_loader = load_dataset(args, split, hml_mode='eval', use_global=args.global_3d, wo_obj_motion=args.wo_obj_motion)


    num_actions = gen_loader.dataset.num_actions


    print("creating affordance model and diffusion ...")
    # NOTE: Hard-coded affordance model

    afford_model, afford_diffusion = load_model(args, gen_loader, dist_util.dev(), 
                                           ModelClass=AffordEstimation, DiffusionClass=AffordDiffusion,
                                           model_path=args.afford_model_path, diff_steps=500)



    print("Creating motion model and diffusion...")
    motion_model, motion_diffusion = load_model(args, gen_loader, dist_util.dev(), ModelClass=used_model,
                                         DiffusionClass=LocalMotionDiffusion, model_path=args.model_path, diff_steps=1000)


    model_dict = {"motion": motion_model, "afford":afford_model}
    diffusion_dict = {"motion": motion_diffusion, "afford": afford_diffusion}

    eval_motion_loaders = {
        ################
        ## behave3D Dataset##
        ################
        'vald': lambda seed: get_mdm_loader_cond(
            model_dict,
            diffusion_dict,
            args.batch_size,
            gen_loader,
            mm_num_samples, 
            mm_num_repeats, 
            gt_loader.dataset.opt.max_motion_length, 
            num_samples_limit, 
            args.guidance_param,
            seed=seed,
            save_dir=save_dir,
            afford_save_dir=afford_save_dir,
            skip_first_stage=args.skip_first_stage,
            wo_obj_motion=args.wo_obj_motion
        )
    }

    eval_wrapper = EvaluatorMDMWrapper(args.dataset, args.global_3d, dist_util.dev())
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm, wo_obj_motion=args.wo_obj_motion)



import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    parser.add_argument("--obj_mesh_path", type=str, default='/work/vig/xiaogangp/codes/hoi-motion_pretrained/object_mesh')
    params = parser.parse_args()

    assert params.input_path.endswith('.mp4')
    parsed_name = os.path.basename(params.input_path).replace('.mp4', '').replace('sample', '').replace('rep', '')
    sample_i, rep_i = [int(e) for e in parsed_name.split('_')]
    npy_path = os.path.join(os.path.dirname(params.input_path), 'results.npy')
    out_npy_path = params.input_path.replace('.mp4', '_smpl_params.npy')
    out_obj_npy_path = params.input_path.replace('.mp4', '_obj_params.npy')
    assert os.path.exists(npy_path)
    results_dir = params.input_path.replace('.mp4', '_obj')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir) 

    # object
    npy2obj_object = vis_utils.npy2obj_object(npy_path, params.obj_mesh_path, sample_i, rep_i,
                                       device=params.device, cuda=params.cuda, if_color=True)

    # # human
    npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                device=params.device, cuda=params.cuda, if_color=True)

    # print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    # for frame_i in tqdm(range(npy2obj.real_num_frames)):
    #     npy2obj.save_ply(os.path.join(results_dir, 'frame{:03d}.ply'.format(frame_i)), frame_i)
    #     npy2obj_object.save_ply(os.path.join(results_dir, 'obj_frame{:03d}.ply'.format(frame_i)), sample_i, frame_i)

    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    npy2obj.save_npy(out_npy_path)
    npy2obj_object.save_npy(out_obj_npy_path)


# blender -b -noaudio --python render.py -- --cfg=./configs/render.yaml --dir=../save/contact_lamda_0.1/samples_contact_000020025_seed10/ --mode=sequence --joint_type=HumanML3D
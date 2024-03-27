
# HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions using Diffusion Models



![](./assets/teaser.png)

<p align="center">
  <a href='https://arxiv.org/abs/2312.06553'>
    <img src='https://img.shields.io/badge/Arxiv-2312.06553-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2312.06553.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href='https://neu-vi.github.io/HOI-Diff/'>
  <img src='https://img.shields.io/badge/Project-Page-orange?style=flat&logo=Google%20chrome&logoColor=orange'></a>
  <!-- <a href='https://youtu.be/0a0ZYJgzdWE'>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a> -->
  <a href='https://github.com/neu-vi/HOI-Diff'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=neu-vi.HOI-Diff&left_color=gray&right_color=blue">
  </a>
</p>


<p align="center">
<!-- <h1 align="center">InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion</h1> -->
<strong>HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions using Diffusion Models</strong></h1>
   <p align="center">
    <a href='https://xiaogangpeng.github.io' target='_blank'>Xiaogang Peng*</a>&emsp;
    <a href='https://ymingxie.github.io' target='_blank'>Yiming Xie*</a>&emsp;
    <a href='http://zizhao.me/' target='_blank'>Zizhao Wu</a>&emsp;
    <a href='https://varunjampani.github.io/' target='_blank'>Varun Jampani</a>&emsp;
    <a href='https://deqings.github.io/' target='_blank'>Deqing Sun</a>&emsp;
    <a href='https://jianghz.me/' target='_blank'>Huaizu Jiang</a>&emsp;
    <br>
    Northeastern University &emsp; Hangzhou Dianzi University &emsp;
    Stability AI &emsp; Google Research
    <br>
    arXiv 2023
  </p>
</p>

## 💻 Demo
![](./assets/demo.gif)


## 📜 TODO List
- [x] Release the dataset preparation and annotations.
- [x] Release the main codes for implementation.
- [ ] Release the evaluation codes and the pretrained models.
- [ ] Release the demo video.

## 📥 Data Preparation

For more information about the implementation, see [README](utils/README.md).

## ⚙️ Quick Start
<details>
  <summary><b>Setup and download</b></summary>



### 1. Setup environment
Install ffmpeg (if not already installed):

```
sudo apt update
sudo apt install ffmpeg
```

Setup conda env:
```
conda env create -f environment.yml
conda activate t2hoi

python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```


Download dependencies:
```
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2hoi_evaluators.sh  
```

Pleas follow [this](https://github.com/erikwijmans/Pointnet2_PyTorch) to install PointNet++.

### 2. Download Pre-trained model
`MDM:` Before your training, please download the pre-trained model [here](https://drive.google.com/file/d/1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821/view?pli=1), then unzip and place them in ./checkpoints/.

`HOI-DM and APDM:` 
Release soon!

### 3. Train your APDM
```
python -m train.train_affordance --save_dir ./save/afford_pred --dataset behave --save_interval 1000 --num_steps 20000 --batch_size 32 --diffusion_steps 500
```

### 4. Train your HOI-DM
```
python -m train.hoi_diff --save_dir ./save/behave_enc_512 --dataset behave --save_interval 1000 --num_steps 20000 --arch trans_enc --batch_size 32
```

### 5. HOIs Synthesis

Generate from test set prompts
```
python -m sample.local_generate_obj --model_path ./save/behave_enc_512/model000020000.pt --num_samples 10 --num_repetitions 1 --motion_length 10 --multi_backbone_split 4 --guidance
```
Generate from your text file
```
python -m sample.local_generate_obj --model_path ./save/behave_enc_512/model000020000.pt --num_samples 10 --num_repetitions 1 --motion_length 10 --multi_backbone_split 4 --guidance
```

<!-- ### 6. Evaluate
```
python -m eval.eval_behave --model_path ./save/behave_enc_512/model000020000.pt  --guidance --comment eval_behave
``` -->

</details>

## Visualization
<details>
<summary><b> Render SMPL mesh</b></summary>

To create SMPL mesh per frame run:

```shell
python -m visualize.render_mesh --input_path /path/to/mp4/stick/figure/file
```

**This script outputs: [YOUR_NPY_FOLDER]**
* `sample##_rep##_smpl_params.npy` - SMPL parameters (human_motion, thetas, root translations, human_vertices and human_faces)
* `sample##_rep##_obj_params.npy` - SMPL parameters (object_motion, object_vertices and object_faces)

**Notes:**
* This script is running [SMPLify](https://smplify.is.tue.mpg.de/) and needs GPU as well (can be specified with the `--device` flag).
* **Important** - Do not change the original `.mp4` path before running the script.

### 1. Set up blender - WIP

Refer to [TEMOS-Rendering motions](https://github.com/Mathux/TEMOS) for blender setup, then install the following dependencies.

```
YOUR_BLENDER_PYTHON_PATH/python -m pip install -r prepare/requirements_render.txt
```
### 2. Render SMPL meshes

Run the following command to render SMPL using blender:

```
YOUR_BLENDER_PATH/blender --background --python render.py -- --cfg=./configs/render.yaml --dir=YOUR_NPY_FOLDER --mode=video --joint_type=HumanML3D
```

optional parameters:

- `--mode=video`: render mp4 video
- `--mode=sequence`: render the whole motion in a png image.
</details>


## Acknowledgments
This code is standing on the shoulders of giants. We want to thank the following contributors that our code is based on:

[BEHAVE](https://github.com/xiexh20/behave-dataset), [MLD](https://github.com/ChenFengYe/motion-latent-diffusion), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [GMD](https://github.com/korrawe/guided-motion-diffusion), [guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi).

## 🤝 Citation
If you find this repository useful for your work, please consider citing it as follows:
```bibtex
@article{peng2023hoi,
  title={HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions using Diffusion Models},
  author={Peng, Xiaogang and Xie, Yiming and Wu, Zizhao and Jampani, Varun and Sun, Deqing and Jiang, Huaizu},
  journal={arXiv preprint arXiv:2312.06553},
  year={2023}
}
```

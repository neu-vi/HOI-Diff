# BEHAVE dataset for text-driven 3D human-object interaction synthesis 
[[ArXiv]](https://arxiv.org/abs/2312.06553) [[Project Page]](https://neu-vi.github.io/HOI-Diff/)


The dataset is a simplified version where raw BEHAVE data has been processed for local human pose using the HumanML3D format, while the object pose has been saved in the global pose space.

## Download
The processed data and annotations can be downloaded [here](https://xxx).


## Dataset Structure
After unzip the dataset, you can find four subfolders: `affordance_data`, `local_pose_rep`, `object_mesh`,  `object_sample`. The summary of each folder is described below:
```
affordance_data: human contact labels, object cotact positions and object state
local_pose_rep: human-object interaction sequences
objects_mesh: 3D scans of the 20 objects
objects_sample: downsampling idxs for 20 objects
split.json: train and test split
```
We discuss details of each folder next:

**local_pose_rep**: This folder stores the human and object pose sequences with text descriptions.

```
DATASET_PATH
|--local_pose_rep          
|----new_joint_vecs_local       # 263-dim local pose vectors
|----new_joint_local   # 22 skeleton joints' positions 
|----texts             # text descriptions
|----Mean_local.npy      
|----Std_local.npy
|----....     
```

**affordance_data**: This folder stores the affordance data for humans and objects, including 8-dim human contact labels, 1-dim object state and two object contact positions of 6-dim.
```
DATASET_PATH
|--affordance_data  
|----sample_name.npy  # affordance data for contact information and object state

```

**object_mesh**: This folder provides the scans of our template objects. 
```
DATASET_PATH
|--object_mesh
|----object_name
|------object_name.jpg  # one photo of the object
|------object_name.obj  # reconstructed 3D scan of the object
|------object_name.obj.mtl  # mesh material property
|------object_name_tex.jpg  # mesh texture
|------object_name_fxxx.ply  # simplified object mesh 
```

**object_sample**: This folder provides the downsampling idxs of the object mesh vertices.
```
DATASET_PATH
|--object_sample
|----sample_name.npy   #  each sample contains 512 downsampling idxs
```

**split.json**: this file provides the official train and test split for the dataset. The split is based on sequence name. In total there are 231 sequences for training and 90 sequences for testing. 


## Example Dataloader
For the annotated dataset, here we provide the example dataloader codes `dataset.py`, which is similar to the HumanML3D dataloader.






## Citation
If you use this data and our annotations, please cite:
```bibtex
@inproceedings{bhatnagar22behave,
  title={Behave: Dataset and method for tracking human object interactions},
  author={Bhatnagar, Bharat Lal and Xie, Xianghui and Petrov, Ilya A and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15935--15946},
  year={2022}
}
```

```
@misc{peng2023hoidiff,
        title={HOI-Diff: Text-Driven Synthesis of 3D Human-Object Interactions using Diffusion Models}, 
        author={Xiaogang Peng and Yiming Xie and Zizhao Wu and Varun Jampani and Deqing Sun and Huaizu Jiang},
        year={2023},
        eprint={2312.06553},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
}
```
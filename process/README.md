

## Data Preparation

We provided the script to convert BEHAVE and OMOMO data into [HumanML3D](https://github.com/EricGuo5513/HumanML3D) format. The human pose is represented in the local space, while the object pose is in the global space.

Firstly, we need the SMPL-H and SMPL-X body model, so please kindly download the latest version (v1.2) of SMPL-H and SMPL-X from the official [website](https://mano.is.tue.mpg.de/), and place it in ./body_models and organize them like this:
```
./body_models/
|--smplh
    |----SMPLH_FEMALE.npz
    |----SMPLH_MALE.npz
    |----SMPLH_NEUTRAL.npz
|--smplx
    |----SMPLX_FEMALE.pkl
    |----SMPLX_MALE.pkl
    |----SMPLX_NEUTRAL.pkl
```

## Preprocess BEHAVE and OMOMO data from scratch:  

First, download BEHAVE official data of SMPLH and object parameters (30fps) from [here](https://virtualhumans.mpi-inf.mpg.de/behave/license.html) , unzip and place them into ./dataset/raw_data/, which would be like this: 
```
│── behave_raw
│   ├── sequence_name
│   │   ├── info.json
│   │   ├── object_fit_all.npz # object's pose sequences
│   │   ├── smpl_fit_all.npz # human's pose sequences
│   └── ...	
│── behave_objects
│   ├── backpack
│   │   ├── backpack.obj
│   │   └── ...
│   └── ...	
```


Then download OMOMO official data from their repo [here](https://drive.google.com/file/d/1tZVqLB7II0whI-Qjz-z-AU3ponSEyAmm/view?usp=sharing) , unzip and place them into ./dataset/raw_data/, which would be like this: 
```
│── omomo_raw
│   ├── train_diffusion_manip_seq_joints24.p
│   ├── test_diffusion_manip_seq_joints24.p
│   └── ...	
│── omomo_objects   
│   ├── clothesstand_cleaned_simplified.obj
│   └── floorlamp_cleaned_simplified.obj
│   └── ...	
```



Run `python process/process_behave.py` to split motion sequences based on our manual annotations. The processed data will be exported to **./dataset/behave_t2m/sequences**, while the corresponding object meshes and down-sampled object points will be saved in **./dataset/behave_t2m/object_mesh**.

Next, run `python process/process_omomo.py` to generate new motion data for the OMOMO dataset. The output motion sequences will be saved in **./dataset/omomo_t2m/sequences**, and the object meshes and down-sampled object points will be saved in **./dataset/omomo_t2m/object_mesh**.

Then, execute `python process/motion_representation_263.py --dataset behave / omomo` to canonicalize the 22 human SMPL joints and the object, and to convert the human joint data into the HumanML3D format for BEHAVE and OMOMO, respectively.

Afterward, run `python process/calc_mean_std.py --dataset behave / omomo` to compute the mean and standard deviation for each dataset.

Finally, execute `python process/get_affordance.py --dataset behave / omomo` to obtain the affordance data, including binary human contact labels and their corresponding contact positions on the object surface for both datasets.


## Dataset Structure
After processed the data, you will have five subfolders: `affordance`, `sequences`, `sequences_263_rep`, `texts`, `object_mesh`,  `sample_objids`. Please organize them as follows:
```
.dataset/behave_t2m/
|--affordance      # human contact labels, object cotact positions
|--sequences_263_rep  # human-object interaction sequences
|--sequences       # 22 human joints, 6-Dof object poses and SMPLH/SMPLX parameters
|--objects_mesh          # scanned mesh of object mesh, bps feature, downsampled object points
|--texts                 # text descriptions
|--split.json            # train and test split
|--Mean_local.npy      
|--Std_local.npy
```


```
.dataset/omomo_t2m/
|--affordance      # human contact labels, object cotact positions
|--sequences_263_rep  # human-object interaction sequences
|--sequences       # 22 human joints, 6-Dof object poses and SMPLH/SMPLX parameters
|--objects_mesh          # scanned mesh of object mesh, bps feature, downsampled object points
|--texts                 # text descriptions
|--split.json            # train and test split
|--Mean_local.npy      
|--Std_local.npy
```


<!-- We discuss details of each folder next:

**new_joint_vecs_local**: This folder stores the HOIs data for humans and objects, including 263-dim pose representation and 6-dim object pose representation.
```
.dataset/behave_t2m/
|--new_joint_vecs_local  
|----sequence_name.npy 
```


**affordance_data**: This folder stores the affordance data for humans and objects, including 8-dim binary human contact labels (8x1) for 8 primary contacting joints, and the corresponding 8-dim object contact positions (8x3). 

```
.dataset/behave_t2m/
|--affordance_data  
|----contact_{sequence_name}.npy  # affordance data for contact information
```

**object_mesh**: This folder provides the scans of our template objects. 
```
.dataset/behave_t2m/
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
.dataset/behave_t2m/
|--object_sample
|----sequence_name.npy   #  each sample contains 512 downsampling idxs
```

**split.json**: this file provides the official train and test split for the dataset. The split is based on sequence name. The splited information is stored in the `train.txt` and `test.txt`. -->











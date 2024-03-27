

## Data Preparation

We have prepocessed the BEHAVE data into [HumanML3D](https://github.com/EricGuo5513/HumanML3D) format. The human pose is represented in the local space, while the object pose is in the global space.

## Download
The preprocessed motion data and annotations can be downloaded from [here](https://drive.google.com/file/d/1w7IRaMMhdU2PM1Dk4nkfKAFbXTf4iPZA/view?usp=sharing).


## Dataset Structure
After unzip the data, you can find five subfolders: `affordance_data`, `new_joint_vecs_local`, `new_joint_local`, `texts`, `object_mesh`,  `object_sample`. Please organize them as follows:
```
.dataset/behave_t2m/
|--affordance_data       # human contact labels, object cotact positions and object state
|--new_joint_vecs_local  # human-object interaction sequences
|--new_joint_local       # human joint sequences in global space
|--objects_mesh          # 3D scans of the 20 objects
|--objects_sample        # downsampling idxs for 20 objects
|--texts                 # text descriptions
|--split.json            # train and test split
|--Mean_local.npy      
|--Std_local.npy
|--test.txt     
|--train.txt
```
We discuss details of each folder next:

**new_joint_vecs_local**: This folder stores the HOIs data for humans and objects, including 263-dim pose representation and 6-dim object pose representation.
```
.dataset/behave_t2m/
|--new_joint_vecs_local  
|----sequence_name.npy 

```


**affordance_data**: This folder stores the affordance data for humans and objects, including 8-dim human contact labels, 1-dim object state and two object contact positions of 6-dim.
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

**split.json**: this file provides the official train and test split for the dataset. The split is based on sequence name. The splited information is stored in the `train.txt` and `test.txt`.

> Utilizing the SMPL-H body model, kindly download the latest version (v1.2) from the official [website](https://mano.is.tue.mpg.de/), and place it in a suitable directory.


### Preprocess BEHAVE data from scratch [Optional]: 
If you want to preprocess motion data from scratch, you could download official data of SMPL and object parameters (30fps) from [here](https://virtualhumans.mpi-inf.mpg.de/behave/license.html) , unzip and place them into ./dataset/behave-30fps-params/, which would be like this: 
```
./data/behave-30fps-params/
|--sequence_name
|----info.json
|----object_fit_all.npz # object's pose sequences
|----smpl_fit_all.npz # human's pose sequences
```
Run  `bash prepare/process_behave_raw.sh` to split motion sequeces based on our manual annonation. The splited data will be output in the ./dataset/raw_behave. Then `python utils/raw_pose_processing_behave.py` to output processed data in folder ./dataset/joints_behave. At last, just like preprocessing for HumanML3D, run `motion_representation.py` and `cal_mean_variance.py` to convert data into HumanML3D format and calculate data std and mean, respectively.










## Data Preparation

We have prepocessed the BEHAVE data into [HumanML3D](https://github.com/EricGuo5513/HumanML3D) format. The human pose is represented in the local space, while the object pose is in the global space.

## Download
The motion data and annotations can be downloaded from [here](https://drive.google.com/file/d/168EPBHlzUZidJG-xaE0YZ6k4fByGTGo4/view?usp=sharing).


## Dataset Structure
After unzip the data, you can find four subfolders: `affordance_data`, `local_pose_rep`, `object_mesh`,  `object_sample`. Please organize them as follows:
```
.dataset/behave/
|--affordance_data   # human contact labels, object cotact positions and object state
|--local_pose_rep   # human-object interaction sequences
|--objects_mesh   # 3D scans of the 20 objects
|--objects_sample   # downsampling idxs for 20 objects
|--split.json   # train and test split
```
We discuss details of each folder next:

**local_pose_rep**: This folder stores the human and object pose sequences with text descriptions.

```
.dataset/behave/
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
.dataset/behave/
|--affordance_data  
|----sequence_name.npy  # affordance data for contact information and object state

```

**object_mesh**: This folder provides the scans of our template objects. 
```
.dataset/behave/
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
.dataset/behave/
|--object_sample
|----sequence_name.npy   #  each sample contains 512 downsampling idxs
```

**split.json**: this file provides the official train and test split for the dataset. The split is based on sequence name. The splited information is stored in the `train.txt` and `test.txt`.

> Utilizing the SMPL-H body model, kindly download the latest version (v1.2) from the official [website](https://mano.is.tue.mpg.de/), and place it in a suitable directory.


### Preprocess data from scratch [Optional]: 
If you want to preprocess motion data from scratch, you could follow [interdiff](https://github.com/Sirui-Xu/InterDiff/blob/main/interdiff/README.md) to download original behave data and put motion data into ./dataset/raw_behave/, which would be like this: 
```
./dataset/raw_behave/
|--sequence_name
|----object_fit_all.npz # object's pose sequences
|----smpl_fit_all.npz # human's pose sequences
```
Then run `raw_pose_processing_behave.py` to output processed data in folder ./dataset/joints_behave. At last, just like preprocessing for HumanML3D, run `motion_representation.py` and `cal_mean_variance.py` to convert data into HumanML3D format and calculate data std and mean, respectively.




## 🚀  Data Loading 
To load the motion and text labels, here we provide the example dataloader `.dataset/dataset.py`, which is similar to the HumanML3D dataloader. Notably, `Text2AffordDataset` is for affordance estimation and `Text2MotionDatasetV2` is for HOI generation.






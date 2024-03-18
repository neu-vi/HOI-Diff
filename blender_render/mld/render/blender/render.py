import math
import os
import sys

import bpy
import numpy as np

from .camera import Camera
from .floor import get_trajectory, plot_floor, show_traj
from .sampler import get_frameidx
from .scene import setup_scene  # noqa
from .tools import delete_objs, load_numpy_vertices_into_blender, mesh_detect
from .vertices import prepare_vertices


def prune_begin_end(data, perc):
    to_remove = int(len(data)*perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=True, write_still=True)



def render(npydata, objnpydata, h_data_path, o_data_path, frames_folder, *, mode, gt=False,
           exact_frame=None, num=8, downsample=True,
           canonicalize=True, always_on_floor=False, denoising=True,
           oldrender=True,jointstype="mmm", res="high", init=True,
           accelerator='gpu',device=[0]):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res, denoising=denoising, oldrender=oldrender,accelerator=accelerator,device=device)

    is_mesh = mesh_detect(npydata)

    # npydata = npydata[::10]
    # center = np.mean(objnpydata, 1)[:,np.newaxis, :]
    # objnpydata[:,:,1] -= 0.1
    # objnpydata[:] = objnpydata[100:101]

    
    # Put everything in this folder
    if mode == "video":
        if always_on_floor:
            frames_folder += "_of"
        os.makedirs(frames_folder, exist_ok=True)
        # if it is a mesh, it is already downsampled
        if downsample and not is_mesh:
            npydata = npydata[::8]
    elif mode == "sequence":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}{ext}"

    elif mode == "frame":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}_{exact_frame}{ext}"

    # remove X% of begining and end
    # as it is almost always static
    # in this part
    if mode == "sequence":
        perc = 0.2
        npydata = prune_begin_end(npydata, perc)
        objnpydata = prune_begin_end(objnpydata, perc)

    if is_mesh:
        from .meshes import Meshes
        data = Meshes(npydata, objnpydata, h_data_path, o_data_path, gt=gt, mode=mode,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor)
    else:
        from .joints import Joints
        data = Joints(npydata, gt=gt, mode=mode,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor,
                      jointstype=jointstype)
    

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    show_traj(data.trajectory)



    # Create a floor
    all_data = np.concatenate([data.data, data.obj_data], 1)
    plot_floor(all_data, big_plane=False)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh)

    frameidx = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidx)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_human_names = []
    imported_obj_names = []
    imported_h_contact_names = []
    imported_o_contact_names = []
    for index, frameidx in enumerate(frameidx):
        if mode == "sequence":
            if nframes_to_render == 1:
                frac = index / nframes_to_render + 1
            else:
                frac = index / (nframes_to_render-1)
            mat, mat2 = data.get_sequence_mat(frac)
        else:
            mat = data.mat
            mat2 = data.mat2
            camera.update(data.get_root(frameidx))

        islast = index == (nframes_to_render-1)


    


        human_name, _ = data.load_in_blender(frameidx, mat)
        obj_name, _ = data.load_obj_in_blender(frameidx, mat2)


        name = f"{str(index).zfill(4)}"

        if mode == "video":
            path = os.path.join(frames_folder, f"frame_{name}.png")
        else:
            path = img_path

        if mode == "sequence":
            imported_human_names.extend(human_name)
            imported_obj_names.extend(obj_name)

        elif mode == "frame":
            camera.update(data.get_root(frameidx))

        if mode != "sequence" or islast:
            render_current_frame(path)
            delete_objs(human_name)
            delete_objs(obj_name)


    # bpy.ops.wm.save_as_mainfile(filepath="/Users/mathis/TEMOS_github/male_line_test.blend")
    # exit()

    # remove every object created
    delete_objs(imported_human_names)
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder", "Sphere"])

    if mode == "video":
        return frames_folder
    else:
        return img_path

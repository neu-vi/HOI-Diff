import numpy as np
import bpy

from .materials import body_material

from .materials import obj_material

from .materials import floor_mat
import matplotlib
# green
# GT_SMPL = body_material(0.009, 0.214, 0.029)
# GT_SMPL = body_material(0.035, 0.415, 0.122)
GT_SMPL = body_material(0.274, 0.674, 0.792)


# blue
GEN_SMPL = body_material(0.022, 0.129, 0.439)
# Blues => cmap(0.87)
# GEN_SMPL = body_material(0.035, 0.322, 0.615)
# Oranges => cmap(0.87)

# (0.658, 0.214, 0.0114) ORANGE 

Orange_mat = body_material(0.658, 0.214, 0.0114) 
Blue_mat = body_material(0.022, 0.129, 0.439)


GEN_SMPL = body_material(0.274, 0.674, 0.792)

BLACK_MAT = body_material(0.0, 0.0, 0.0)
WHITE_MAT = body_material(1.0, 1.0, 1.0)

class Meshes:
    def __init__(self, h_data, o_data, h_data_path, o_data_path, *, gt, mode, canonicalize, always_on_floor, oldrender=False, **kwargs):
        

        
        self.faces = np.load(h_data_path, allow_pickle=True).item()['faces']
        self.obj_faces = np.load(o_data_path, allow_pickle=True).item()['faces']


        
        data = prepare_meshes(h_data, canonicalize=canonicalize,
                              always_on_floor=True)
        
        obj_data = prepare_obj_meshes(o_data, canonicalize=canonicalize,
                              always_on_floor=True)

        
        self.data = data
        self.obj_data = obj_data 
        self.mode = mode
        self.oldrender = oldrender

        self.N = len(data)
        self.trajectory = data[:, :, [0, 1]].mean(1)

        # if gt:
        #     self.mat = GT_SMPL
        #     self.mat2 = GEN_SMPL
        # else:

        cmap = matplotlib.cm.get_cmap('Oranges')
        cmap2 = matplotlib.cm.get_cmap('Blues')
        # begin = 0.60
        # end = 0.90
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(end)
        rgbcolor2 = cmap2(end)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        mat2 = obj_material(*rgbcolor2, oldrender=self.oldrender)
        self.mat = mat
        self.mat2 = mat2

    def get_sequence_mat(self, frac):
        import matplotlib
        # cmap = matplotlib.cm.get_cmap('Blues')
        cmap = matplotlib.cm.get_cmap('Oranges')
        cmap2 = matplotlib.cm.get_cmap('Blues')
        # begin = 0.60
        # end = 0.90
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end-begin)*frac)
        rgbcolor2 = cmap2(begin + (end-begin)*frac)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        mat2 = obj_material(*rgbcolor2, oldrender=self.oldrender)
        
        return mat, mat2

        
    

    def get_root(self, index):
        return self.data[index].mean(0)

    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index, mat):
        
        h_contact_names = []
        vertices = self.data[index]
        faces = self.faces
        name = f"{str(index).zfill(4)}"

        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)
        

        return name, None
    
    def load_obj_in_blender(self, index, mat):
        vertices = self.obj_data[index]
        name = f"obj_{str(index).zfill(4)}"
        faces = self.obj_faces
        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)
        
        
        return name, None

    def __len__(self):
        return self.N


def prepare_meshes(data, canonicalize=True, always_on_floor=False):
    if canonicalize:
        print("No canonicalization for now")

    # fitted mesh do not need fixing axis
    # # fix axis
    # data[..., 1] = - data[..., 1]
    # data[..., 0] = - data[..., 0]

    # Swap axis (gravity=Z instead of Y)
    data = data[..., [2, 0, 1]]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data

def prepare_obj_meshes(data, canonicalize=True, always_on_floor=False):
    if canonicalize:
        print("No canonicalization for now")

    # fitted mesh do not need fixing axis
    # # fix axis
    # data[..., 1] = - data[..., 1]
    # data[..., 0] = - data[..., 0]

    # Swap axis (gravity=Z instead of Y)
    data = data[..., [2, 0, 1]]

    # # Remove the floor
    # data[..., 2] -= data[..., 2].min()

    # # Put all the body on the floor
    # if always_on_floor:
    #     data[..., 2] -= data[..., 2].min(1)[:, None]

    return data

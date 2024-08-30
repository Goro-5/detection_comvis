import trimesh
import pyrender
import math
import edge_generate as eg
import numpy as np

def view_mesh_pyrender(file_path, pattern, r):
    mesh = trimesh.load(file_path)
    eg.apply_transform(mesh, pattern, r)
    
    # pyrenderシーンの作成
    scene = pyrender.Scene()
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.5, 0.5, 0.5, 1.0],
        metallicFactor=0.7,
        roughnessFactor=0.7
    )
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(mesh)
    
    # カメラの設定
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1920 / 1680)
    camera_pose = np.array([
        [1,  0,  0, 0],
        [0,  1,  0, 0],
        [0,  0, -1, 20],
        [0,  0,  0, 1]
    ])
    scene.add(camera, pose=camera_pose)
    
    # ライトの追加
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=camera_pose)
    
    # ビューアで表示
    pyrender.Viewer(scene, use_raymond_lighting=False)
    
# オブジェクトを表示する
#view_mesh_pyrender("T:/Goro/ComputerVision/joints_data/PS-10SH.obj", "FRONT", 0)
eg.viewer("PS-10SH","FRONT",0,320,100,[0.1,0.1,0.1])
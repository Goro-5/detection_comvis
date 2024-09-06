import trimesh
import pyvista as pv
import numpy as np
from PIL import Image

# OBJファイルとMTLファイルの読み込み
mesh = trimesh.load('T:\\Goro\\ComputerVision\\joints_data\\PS-10SH.obj')

# メッシュの頂点と面を取得
vertices = mesh.vertices
faces = mesh.faces

# PyVistaのフォーマットに合わせるための面データの変換
faces_expanded = np.c_[np.full(len(faces), 3), faces].flatten()

# PyVistaのPolyDataオブジェクトを作成
polydata = pv.PolyData(vertices, faces_expanded)

# テクスチャの読み込み
mtl_path = 'T:\\Goro\\ComputerVision\\joints_data\\PS-10SH.mtl'
texture_path = None

with open(mtl_path, 'r') as file:
    for line in file:
        if line.startswith('map_Kd'):
            texture_path = line.split()[1]
            break

if texture_path:
    # テクスチャ画像の読み込み
    texture_image = Image.open('T:\\Goro\\ComputerVision\\joints_data\\' + texture_path)
    texture_image = np.array(texture_image) / 255.0  # 0-1の範囲にスケール

    # テクスチャ座標の取得
    if hasattr(mesh.visual, 'uv'):
        uv = mesh.visual.uv
    else:
        uv = None

    # テクスチャをPolyDataに適用
    if uv is not None:
        polydata.active_t_coords = uv
        texture = pv.numpy_to_texture(texture_image)
        polydata.texture_map_to_plane(inplace=True, use_bounds=True)
    else:
        print("UV座標が見つかりません。")

# プロットの設定
plotter = pv.Plotter()
if uv is not None and texture_path:
    plotter.add_mesh(polydata, texture=texture)
else:
    plotter.add_mesh(polydata, color='white')  # テクスチャがない場合のデフォルト色

plotter.show()

import numpy as np
import trimesh
import pyrender
import math
import os
from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import *
from OpenGL.GLUT import *
from pyrender.viewer import Viewer as PygletViewer
from OpenGL.GLU import *
import cv2
import random


def load_obj(file_path):
    return trimesh.load(file_path)



def apply_transform(mesh, pattern, r):
    if pattern == "FRONT":
        rotation = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
    elif pattern == "BACK":
        rotation = trimesh.transformations.rotation_matrix(math.radians(-90), [0, 1, 0])
    elif pattern == "SIDE":
        rotation = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        r += 90
    

    r_rotation = trimesh.transformations.rotation_matrix(math.radians(r), [0, 0, 1])
    
    mesh.apply_transform(rotation)
    mesh.apply_transform(r_rotation)
    
    # センタリングと底面を合わせる
    mesh.vertices -= mesh.bounds.mean(axis=0)
    mesh.vertices -= [0, 0, mesh.bounds[0][2]]

def add_mesh_to_scene(scene, mesh):
    material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.5, 0.5, 0.5, 1.0],  # ステンレスの色に近いグレー
    metallicFactor=0.8,  # 完全に金属的
    roughnessFactor=0.4  # ステンレスの光沢を反映
)
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(mesh)

def create_camera(z, yfov, znear=1, zfar=1000, x=0, y=0):
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1920 / 1680, znear=znear, zfar=zfar)
    camera_pose = np.array([
        [1,  0,  0, x],   # x軸の方向
        [0,  1,  0, y],   # y軸の方向
        [0,  0,  1, z],   # z軸の方向と平行移動
        [0,  0,  0, 1]    # 同次座標系の最後の行
    ])
    return camera, camera_pose

def render_scene(scene, camera, camera_pose):
    r = pyrender.OffscreenRenderer(1920, 1680)
    
    # カメラをシーンに追加
    camera_node = scene.add(camera, pose=camera_pose)
    
    # レンダリング
    color, _ = r.render(scene)
    
    # カメラをシーンから削除
    scene.remove_node(camera_node)
    
    return color

def save_image(image, output_path):
    Image.fromarray(image).save(output_path)

def create_scene(bg_color):
    return pyrender.Scene(bg_color=bg_color)

def add_light(scene, intensity, z=1000, x=0,y=0):
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
    light_pose = np.array([
        [1,  0,  0, x],   # x軸の方向
        [0,  1,  0, y],   # y軸の方向
        [0,  0,  1, z],   # z軸の方向と平行移動
        [0,  0,  0, 1]    # 同次座標系の最後の行
    ])
    scene.add(light, pose=light_pose)

def render_image(mesh,z,intensity,bg_color,yfov,x=0,y=0):
    scene = create_scene(bg_color)
    add_mesh_to_scene(scene, mesh)
    add_light(scene, intensity, 10*z)
    add_light(scene,intensity*0.7,8*z,1400,1300)
    add_light(scene,intensity*0.7,8*z,-700,500)
    add_light(scene,intensity*0.7,8*z,100,-2300)
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity*1.5)
    scene.add(dl)
    
    
    # クリッピング平面の設定
    znear = 0.1
    zfar = 10000  # 遠くのオブジェクトもクリップしないように大きな値を設定
    
    camera, camera_pose = create_camera(z, yfov, znear, zfar, x, y)
    image = render_scene(scene, camera, camera_pose)
    return image

def make_and_render(obj_name,pattern,r,distance=320,light_intensity=200000,bg_color=[0,0,0,0],yfov=np.pi/3,x=0,y=0):
    mesh = load_obj(f"T:/Goro/ComputerVision/joints_data/{obj_name}.obj")
    apply_transform(mesh, pattern, r)

    image = render_image(mesh,distance,light_intensity,bg_color,yfov,x,y)
    return image

def edge_detection(image):
    color = np.array(image, dtype=np.uint8)
    gray = cv2.cvtColor(color, cv2.COLOR_BGRA2GRAY)


    # Cannyエッジ検出
    # 低いしきい値と高いしきい値を調整
    low_threshold = 30
    high_threshold = 150
    canny_edges = cv2.Canny(gray, low_threshold, high_threshold)
    return canny_edges

def cropped_image(image):
    color = np.array(image, dtype=np.uint8)
    # グレースケールに変換
    gray = cv2.cvtColor(color, cv2.COLOR_BGRA2GRAY)

    # Cannyエッジ検出を適用
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    # 輪郭の検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:

    # 最大の輪郭を見つける
        max_contour = max(contours, key=cv2.contourArea)

        # 最大の輪郭を囲む最小の水平な長方形を計算
        x, y, w, h = cv2.boundingRect(max_contour)

        # 画像を切り取る
        cropped_img = color[y:y+h, x:x+w]
    else:
        cropped_img = None
    return cropped_img

def resize_edged(img,size=(244,244)):
    # 画像をリサイズ
    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # リサイズされた画像をグレースケールに変換
    resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # エッジ検出を適用
    resized_edges = cv2.Canny(resized_gray, 50, 150)
    return resized_edges


def boundary_XY(image,scale):
    color = np.array(image, dtype=np.uint8)
    gray = cv2.cvtColor(color, cv2.COLOR_BGRA2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    # 輪郭を取得
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    nx, ny = 0, 0

    if len(contours) != 0:

        area_max = 0
        max_x = None
        max_y = None
        max_w = None
        max_h = None

        for cnt in contours:
            # 単純な外接矩形を描画
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > area_max:
                area_max = w * h
                max_x = x
                max_y = y
                max_w = w
                max_h = h
        height, width = gray.shape
        dh = height-max_h
        dw = width-max_w
        nx = round(dh*scale*0.5)
        ny = round(dw*scale*0.5)
        mx = round((max_x-dh/2)*scale)
        my = round((max_y-dw/2)*scale)
        print(f"h:{max_h} w:{max_w}")
        print(f"dh:{dh} dw:{dw}")
        print(f"nx:{nx} ny:{ny}")
    
    return nx, ny, mx, my


def render_grid2(obj_list, pattern_list, r_list, scale,  distance=320, light_intensity=200000, bg_color=[0,0,0,0], yfov=np.pi/3, num=100000):
    n = len(obj_list)
    m = len(pattern_list)
    o = len(r_list)
    std_mtx = np.zeros((n,m,o,2))
    mean_mtx = np.zeros((n,m,o,2))
    n = -1
    for obj_name in obj_list:
        n += 1
        m = -1
        for pattern in pattern_list:
            m += 1
            o = -1
            for r in r_list:
                o += 1
                image = make_and_render(obj_name,pattern,r,distance,light_intensity,bg_color,yfov)
                std_mtx[n][m][o][0], std_mtx[n][m][o][1], mean_mtx[n][m][o][0], mean_mtx[n][m][o][1] = boundary_XY(image,scale)
    for _ in range(0,num):
        nn = random.randint(0,n)
        mm = random.randint(0,m)
        oo = random.randint(0,o)
        std = std_mtx[nn][mm][oo]/3 #3σ区域:97%
        mean = mean_mtx[nn][mm][oo]
        
        obj_name = obj_list[nn]
        pattern = pattern_list[mm]
        r = r_list[oo]
        x = int(np.random.normal(loc=mean[0],scale=std[0]))
        y = int(np.random.normal(loc=mean[1],scale=std[1]))

        image_each = make_and_render(obj_name,pattern,r,distance,light_intensity,bg_color,yfov,x=x,y=y)
        cropped = cropped_image(image_each)
        edge_each = edge_detection(image_each)

        if cropped is not None:
            edge_cropped = resize_edged(cropped)

            # 個別の画像を保存
            output_dir = f"T:/Goro/ComputerVision/code/OpenCV/make_render_edge/joints_edge/{obj_name}/{pattern}"
            os.makedirs(output_dir, exist_ok=True)
            # 既存のPNGファイルの数を数える
            existing_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            file_number = len(existing_files)
            file_name = f"x{x}_y{y}_r{r}_edge{file_number:03d}.png"
            cv2.imwrite(os.path.join(output_dir, file_name),edge_each)

            output_dir = f"T:/Goro/ComputerVision/code/OpenCV/make_render_edge/joints_cropped_edge/{obj_name}/{pattern}"
            os.makedirs(output_dir, exist_ok=True)
            file_name = f"x{x}_y{y}_r{r}_cropped{file_number:03d}.png"
            cv2.imwrite(os.path.join(output_dir, file_name),edge_cropped)

            pil_image = Image.fromarray(image_each)
            output_dir = f"T:/Goro/ComputerVision/code/OpenCV/make_render_edge/joints_vanilla/{obj_name}/{pattern}"
            os.makedirs(output_dir, exist_ok=True)
            file_name = f"x{x}_y{y}_r{r}_vanilla{file_number:03d}.png"
            pil_image.save(os.path.join(output_dir, file_name))
        else:
            print(f"Error:{obj_name}/{pattern}/x{x}_y{y}_r{r}")

def render_grid(obj_name, pattern, r, distances, light_intensities, bg_colors):
    mesh = load_obj(f"T:/Goro/ComputerVision/joints_data/{obj_name}.obj")
    apply_transform(mesh, pattern, r)

    images = []
    for z in distances:
        for intensity in light_intensities:
            for bg_color in bg_colors:
                image = render_image(mesh,z,intensity,bg_color,yfov=np.pi/3)
                
                # パラメータ情報を画像に追加
                pil_image = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.load_default()
                text = f"Distance: {z}, Light: {intensity}, BG: {bg_color}"
                draw.text((10, 10), text, font=font, fill=(255, 255, 255))
                
                images.append(np.array(pil_image))

                # 個別の画像を保存
                output_dir = f"T:/Goro/ComputerVision/code/OpenCV/make_render_edge/{obj_name}/{pattern}"
                os.makedirs(output_dir, exist_ok=True)
                file_name = f"d{z}_l{intensity}_bg{''.join(map(str, bg_color))}_r{r}.png"
                pil_image.save(os.path.join(output_dir, file_name))

    '''# グリッド画像の作成と保存
    rows = len(distances)
    cols = len(light_intensities) * len(bg_colors)
    grid = Image.new('RGB', (1920 * cols, 1680 * rows))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(Image.fromarray(img), (col * 1920, row * 1680))

    grid.save(os.path.join(output_dir, "grid_render.png"))'''

class CustomViewer(PygletViewer):
    def __init__(self, scene, camera_node, **kwargs):
        self.camera_node = camera_node
        self.camera_position = scene.get_pose(camera_node)[:3, 3]
        super().__init__(scene, **kwargs)

    def on_draw(self):
        super().on_draw()
        self._draw_camera_position()

    def _draw_camera_position(self):
        x, y, z = self.camera_position
        camera_position_text = f"Camera Position: ({x:.2f}, {y:.2f}, {z:.2f})"

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self._viewport_size[0], 0, self._viewport_size[1])
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glColor3f(1.0, 0.0, 1.0)
        glRasterPos2f(10, self._viewport_size[1] - 20)
        for ch in camera_position_text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def _update_camera_position(self):
        self.camera_position = self.scene.get_pose(self.camera_node)[:3, 3]

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        self._update_camera_position()

    def on_mouse_scroll(self, x, y, dx, dy):
        super().on_mouse_scroll(x, y, dx, dy)
        self._update_camera_position()



def viewer(obj_name, pattern, r, distance, light_intensity, bg_color):
    mesh = load_obj(f"T:/Goro/ComputerVision/joints_data/{obj_name}.obj")
    apply_transform(mesh, pattern, r)
    scene = create_scene(bg_color)
    add_mesh_to_scene(scene, mesh)
    add_light(scene, light_intensity)
    
    # 焦点距離120mmに対応する視野角を計算
    focal_length = 12
    sensor_height = 24
    yfov = 2 * np.arctan(sensor_height / (2 * focal_length))
    
    # クリッピング平面の設定
    znear = 0.1
    zfar = 10000  # 遠くのオブジェクトもクリップしないように大きな値を設定
    
    camera, camera_pose = create_camera(distance, yfov, znear, zfar)
    camera_node = scene.add(camera, pose=camera_pose)
    glutInit()
    CustomViewer(scene, camera_node, use_raymond_lighting=False)

def main():
    obj_list = ["PS-10SH","PS-18SU","PS-24SU","PS-33SU","TH-10","TH-18","TH-24","TH-33"]
    pattern_list = ["FRONT","BACK","SIDE"]
    r_values = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150] 
    r_values = [0]

    distances = [320]  # カメラ距離
    light_intensities = [200]  # ライト強度
    bg_colors = [[0,0,0,0]]  # 背景色
    for obj_name in obj_list:
        for pattern in pattern_list:
            for r in r_values:
                render_grid(obj_name, pattern, r, distances, light_intensities, bg_colors)


    

if __name__ == "__main__":
    main()

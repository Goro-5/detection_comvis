import sys
import numpy as np
import trimesh
import pyrender
import math
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QOpenGLWidget
from PyQt5.QtCore import QTimer
from OpenGL.GL import *
from OpenGL.GLUT import *

class RenderWidget(QOpenGLWidget):
    def __init__(self, scene, camera_node):
        super().__init__()
        self.scene = scene
        self.camera_node = camera_node
        self.renderer = pyrender.OffscreenRenderer(1920, 1080)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def initializeGL(self):
        glutInit()

    def paintGL(self):
        color, _ = self.renderer.render(self.scene)
        glDrawPixels(1920, 1080, GL_RGB, GL_UNSIGNED_BYTE, color)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

class MainWindow(QWidget):
    def __init__(self, obj_name, pattern, r, distance, light_intensity, bg_color):
        super().__init__()

        self.setWindowTitle("Camera Position Viewer")
        self.setGeometry(100, 100, 1920, 1080)

        layout = QVBoxLayout()
        self.render_widget = None
        self.camera_position_label = QLabel()
        self.camera_position_label.setStyleSheet("QLabel { color : white; }")
        layout.addWidget(self.camera_position_label)
        
        mesh = self.load_obj(f"T:/Goro/ComputerVision/joints_data/{obj_name}.obj")
        self.apply_transform(mesh, pattern, r)
        scene = self.create_scene(bg_color)
        self.add_mesh_to_scene(scene, mesh)
        self.add_light(scene, light_intensity)
        
        focal_length = 120
        sensor_height = 24
        yfov = 2 * np.arctan(sensor_height / (2 * focal_length))
        znear = 0.1
        zfar = 10000
        camera, camera_pose = self.create_camera(distance, yfov, znear, zfar)
        camera_node = scene.add(camera, pose=camera_pose)
        
        self.render_widget = RenderWidget(scene, camera_node)
        layout.addWidget(self.render_widget)
        self.setLayout(layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_position)
        self.timer.start(100)

    def update_camera_position(self):
        camera_pose = self.render_widget.scene.get_pose(self.render_widget.camera_node)
        x, y, z = camera_pose[0, 3], camera_pose[1, 3], camera_pose[2, 3]
        self.camera_position_label.setText(f"Camera Position: ({x:.2f}, {y:.2f}, {z:.2f})")

    def load_obj(self, file_path):
        return trimesh.load(file_path)

    def apply_transform(self, mesh, pattern, r):
        if pattern == "FRONT":
            rotation = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
        elif pattern == "BACK":
            rotation = trimesh.transformations.rotation_matrix(math.radians(-90), [0, 1, 0])
        elif pattern == "SIDE":
            rotation = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])

        r_rotation = trimesh.transformations.rotation_matrix(math.radians(r), [0, 0, 1])

        mesh.apply_transform(rotation)
        mesh.apply_transform(r_rotation)

        # センタリングと底面を合わせる
        mesh.vertices -= mesh.bounds.mean(axis=0)
        mesh.vertices -= [0, 0, mesh.bounds[0][2]]

    def add_mesh_to_scene(self, scene, mesh):
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.5, 0.5, 0.5, 1.0],
            metallicFactor=0.7,
            roughnessFactor=0.3
        )
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh)

    def create_camera(self, distance, yfov, znear, zfar):
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1920 / 1680, znear=znear, zfar=zfar)
        camera_pose = np.array([
            [1,  0,  0, 0],   # x軸の方向
            [0,  1,  0, 0],   # y軸の方向
            [0,  0, -1, distance], # z軸の方向と平行移動
            [0,  0,  0, 1]    # 同次座標系の最後の行
        ])
        return camera, camera_pose

    def create_scene(self, bg_color):
        return pyrender.Scene(ambient_light=bg_color)

    def add_light(self, scene, intensity):
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        scene.add(light, pose=np.eye(4))

def main():
    obj_name = "PS-10SH"
    pattern = "FRONT"
    r = 0
    distance = 35
    light_intensity = 200
    bg_color = [0.1, 0.1, 0.1]

    app = QApplication(sys.argv)
    window = MainWindow(obj_name, pattern, r, distance, light_intensity, bg_color)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

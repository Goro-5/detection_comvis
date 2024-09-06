import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np

# ボタンが押された時の動作
def submit_action(entry_height, entry_size):
    height = entry_height.get()
    size = entry_size.get()

    try:
        # 入力された値をfloatに変換し、メッセージで表示
        height = int(height)
        size = float(size)
        messagebox.showinfo("入力内容", f"高さ: {height} mm, 大きさ: {size} mm")
        return height, size
    except ValueError:
        messagebox.showerror("エラー", "有効な数値を入力してください。")
        return None, None

def get_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けません")
        return None
    while True:
        ret, frame = cap.read()
        # Show camera output on screen
        cv2.imshow("Camera Output", frame)
        cv2.waitKey(1)

        # If QR code is not detected, show warning message on screen
        if not qr_code_detected(frame):
            cv2.putText(frame, "QR code UNDETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Camera Output", frame)
            cv2.waitKey(1)
        # Perform QR code detection on the frame
        # If QR code is detected, break the loop and return the cap
        if qr_code_detected(frame):
            cv2.destroyAllWindows()
            return frame
        # Refresh the cap by releasing and reopening it
        cap.release()
        cap = cv2.VideoCapture(0)

def qr_code_detected(frame):
    qr = cv2.QRCodeDetector()
    data, points, straight_qrcode = qr.detectAndDecode(frame)
    if points is not None:
        return True
    return False

def process(entry_height, entry_size):
    height, size = submit_action(entry_height, entry_size)
    if height is not None and size is not None:
        frame = get_camera()
        focal_length = calculate_focal_length(height, size, frame)
        camera_pixel_height = frame.shape[0]
        camera_pixel_width = frame.shape[1]
        messagebox.showinfo("処理結果", f"焦点距離: {focal_length} mm\nカメラの画素数: {camera_pixel_height} x {camera_pixel_width}")

def calculate_focal_length(height, size, frame):
    # QRコードの検出
    qr = cv2.QRCodeDetector()
    data, points, straight_qrcode = qr.detectAndDecode(frame)
    if points is not None:
        # QRコードの正方形のピクセル上での面積を計算
        qr_area = cv2.contourArea(points)

        # 焦点距離を計算
        focal_length = (size ** 2) * height / qr_area

        return focal_length
        

def input():

    # ウィンドウの設定
    root = tk.Tk()
    root.title("入力フォーム")

    # 高さの入力フィールドと「mm」ラベルを一つのフレームにまとめる
    frame_height = tk.Frame(root)
    label_height = tk.Label(frame_height, text="カメラ高さ:")
    label_height.pack(side=tk.LEFT)
    entry_height = tk.Entry(frame_height)
    entry_height.insert(0, "350")  # 初期値を350に設定
    entry_height.pack(side=tk.LEFT)
    label_height_unit = tk.Label(frame_height, text=" mm")
    label_height_unit.pack(side=tk.LEFT)
    frame_height.pack(pady=5)

    # 大きさの入力フィールドと「mm」ラベルを一つのフレームにまとめる
    frame_size = tk.Frame(root)
    label_size = tk.Label(frame_size, text="QRコード1辺長さ:")
    label_size.pack(side=tk.LEFT)
    entry_size = tk.Entry(frame_size)
    entry_size.insert(0, "40.0")  # 初期値を40.0に設定
    entry_size.pack(side=tk.LEFT)
    label_size_unit = tk.Label(frame_size, text=" mm")
    label_size_unit.pack(side=tk.LEFT)
    frame_size.pack(pady=5)

    # 送信ボタン
    submit_button = tk.Button(root, text="送信", command=lambda: process(entry_height, entry_size))
    submit_button.pack(pady=10)

    # ウィンドウを表示
    root.mainloop()


if __name__ == "__main__":
    input()
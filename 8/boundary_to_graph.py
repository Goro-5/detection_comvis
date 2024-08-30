import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_distances(largest_cnt, box):
    distances = []
    for i, point in enumerate(largest_cnt):
        point = point[0]  # Extract the point from the numpy array
        min_distance = float('inf')
        for box_point in box:
            distance = np.linalg.norm(point - box_point)
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)
    return distances

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Cannot detect camera")
    exit()

# 解像度を設定
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 幅を1920ピクセルに設定
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1680) # 高さを1680ピクセルに設定

plt.ion()  # インタラクティブモードをオンにする
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')
ax.set_xlim(0, 100)  # 仮のX軸の範囲、必要に応じて変更してください
ax.set_ylim(0, 1000)  # 仮のY軸の範囲、必要に応じて変更してください
ax.set_xlabel('Point index')
ax.set_ylabel('Distance to nearest edge')
ax.set_title('Distance from largest contour points to box')

while True:
    ret, frame = capture.read()
    if ret:
        # 画像をグレースケールに変換し、閾値処理で二値化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 輪郭を取得
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            area_max = 0
            largest_cnt = None

            for cnt in contours:
                # 単純な外接矩形を描画
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h > area_max:
                    area_max = w * h
                    largest_cnt = cnt
            
            if largest_cnt is not None:
                cv2.drawContours(frame, [largest_cnt], 0, (255, 0, 0), 3)
                # 物体の回転を考慮した外接矩形を描画
                rect = cv2.minAreaRect(largest_cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

                # 距離を計算
                distances = calculate_distances(largest_cnt, box)

                # グラフを更新
                line.set_data(range(len(distances)), distances)
                ax.set_xlim(0, len(distances) - 1)
                ax.set_ylim(0, max(distances) + 10)  # Y軸の範囲を調整
                plt.draw()
                plt.pause(0.001)  # グラフ更新のための待機時間

        cv2.imshow('frame', frame)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
plt.close('all') 

import cv2
import numpy as np

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Cannot detect camera")
    exit()
# 解像度を設定
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 幅を1920ピクセルに設定
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1680) # 高さを1680ピクセルに設定
while True:
    ret, frame = capture.read()
    if ret:
        img = frame
        # QRコード検出用のインスタンスを作成
        qr = cv2.QRCodeDetector()
        # QRコードを検出
        data, points, straight_qrcode = qr.detectAndDecode(img)
        if points is not None and len(points) > 0:
            points = points.astype(np.int32)
            # QRコードを検出した位置に四角形を描画
            cv2.polylines(img, [points], True, (255, 0, 0), thickness=5)
            print(data)
        cv2.imshow('qrcode_detect', cv2.resize(img, (1920,1680), interpolation=cv2.INTER_AREA))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()

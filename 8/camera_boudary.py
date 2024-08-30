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
        # 画像をグレースケールに変換し、閾値処理で二値化
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        # 輪郭を取得
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]

        area_max = 0
        largest_cnt = None

        for cnt in contours:
            # 単純な外接矩形を描画
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > area_max:
                area_max = w * h
                largest_cnt = cnt
        
        cv2.drawContours(frame, [largest_cnt], 0, (255,0,0), 3)
        # 物体の回転を考慮した外接矩形を描画
        rect = cv2.minAreaRect(largest_cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if np.linalg.norm(box[0]-box[1])>np.linalg.norm(box[1]-box[2]):
            box = box[1:] + box[:1]
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

        qr = cv2.QRCodeDetector()
        # QRコードを検出
        data, points, straight_qrcode = qr.detectAndDecode(frame)
        if points is not None and len(points) > 0:
            points = points.astype(np.int32)
            # QRコードを検出した位置に四角形を描画
            cv2.polylines(frame, [points], True, (255, 0, 0), thickness=5)
            print(data)
        for x in range(len(box)):
            cv2.putText(frame,f"{x}",box[x],fontFace=1,fontScale=10,color=(255,255,255),thickness=2)

    
        cv2.imshow('frame',frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
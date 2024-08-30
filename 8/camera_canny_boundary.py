import cv2
import numpy as np

def hsv_to_bgr(h, s, v):
    bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8),
                       cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, bgr))

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






        # 画像の読み込み
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # ノイズ除去
        kernel = np.ones((10, 10), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # エッジ検出
        edge = cv2.Canny(gray, 100, 200, apertureSize=3)
        # 輪郭を描画するための画像を作成
        img = frame

        

        # 輪郭の取得方法別にループ
        # RETR_LIST: 親要素、子要素が同等に扱われ，いずれも単なる輪郭として取得
        # RETR_EXTERNAL: 最も外側の輪郭のみを取得し、子要素は無視される
        # RETR_CCOMP: 物体の外側の輪郭と内側の輪郭の2つの階層に分類
        # RETR_TREE: 全階層情報を取得
        retr = cv2.RETR_EXTERNAL
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
        # 輪郭を取得
        contours, hierarchy = cv2.findContours(thresh, retr, cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭を描画するための画像のコピーを作成
        copied_img = img.copy()
        # 輪郭毎に色を変更し、輪郭を描画
        for index, contour in enumerate(contours):
            # 輪郭毎にhueを10加算して、描画する色を変更
            color = hsv_to_bgr(10*index, 255, 255)
            # 輪郭を描画
            cv2.drawContours(copied_img, [contour], 0, color, 5)

            cv2.imshow(f'contour_external', copied_img)



        

        cv2.imshow('edge',edge)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
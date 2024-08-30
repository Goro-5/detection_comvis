import numpy as np
import cv2


def resize_edged(img,size=(244,244)):
    # 画像をリサイズ
    resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # リサイズされた画像をグレースケールに変換
    resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # エッジ検出を適用
    resized_edges = cv2.Canny(resized_gray, 320, 500)
    return resized_edges

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
def cropresize(image, size=(244,244), show=False):
    cropped = cropped_image(image)
    if cropped is not None:
        resized = resize_edged(cropped,size)
        if show:
            cv2.imshow("resize",resized)
    else:
        resized=None
    return resized

image = cv2.imread("T:\Goro\ComputerVision\CamBookRaw\IMG_0007.png")
cropped = cropresize(image,show=True)
cv2.waitKey(0)
cv2.destroyAllWindows()

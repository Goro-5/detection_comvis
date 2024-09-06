import cv2
import numpy as np




def old_func():

    while True:
        ret, frame = capture.read()
        if ret:
            # 画像をグレースケールに変換し、閾値処理で二値化
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            cv2.imshow("thresh",thresh)

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

            cv2.imshow('frame',frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
        else:
            break


def automatic_canny_threshold(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper

def new_func():
    while True:
        ret, frame = capture.read()
        if ret:
            # グレースケールに変換する
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 画像をぼかす（ノイズを減らすため）
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # 自動閾値を用いたCannyエッジ検出
            lower, upper = automatic_canny_threshold(blurred)
            edges = cv2.Canny(blurred, lower, upper)

            # 輪郭と階層情報を抽出する
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 外側の輪郭のみを選択する
            outer_contours = []
            for i in range(len(contours)):
                # 親輪郭がない（hierarchy[0][i][3] == -1）ものが外側の輪郭
                if hierarchy[0][i][3] == -1:
                    outer_contours.append(contours[i])

            # 最大の外側の輪郭を見つける
            if outer_contours:
                largest_contour = max(outer_contours, key=cv2.contourArea)
                # 最大の輪郭を描画する
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

            cv2.imshow('frame', frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
        else:
            break

def hed_func():
    while True:
        ret, frame =capture.read()
        if ret:
        # HEDモデルの読み込み
            prototxt_path = "8\hed-edge-detector-master\hed-edge-detector-master\deploy.prototxt"
            caffemodel_path = "8\hed-edge-detector-master\hed-edge-detector-master\hed_pretrained_bsds.caffemodel"
            net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

            
            (h, w) = frame.shape[:2]

            # 画像をHEDモデルの入力サイズにリサイズ
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(w, h), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)

            # HEDモデルに入力を設定
            net.setInput(blob)

            # エッジマップを取得
            hed = net.forward()

            # エッジマップを画像サイズにリサイズ
            hed = cv2.resize(hed[0, 0], (w, h))

            # エッジマップを正規化
            hed = (255 * hed).astype("uint8")

            # 結果を表示
            cv2.imshow('Original Image', frame)
            cv2.imshow('HED Edge Map', hed)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
        else:
            break
if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot detect camera")
        exit()

    '''# 解像度を設定
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 幅を1920ピクセルに設定
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1680) # 高さを1680ピクセルに設定'''

    old_func()
    capture.release()
    cv2.destroyAllWindows()
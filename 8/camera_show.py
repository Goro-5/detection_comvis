import cv2

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
        cv2.imshow('frame',frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
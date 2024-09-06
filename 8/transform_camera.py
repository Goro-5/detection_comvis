import cv2
import transform
import os

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Cannot detect camera")
    exit()

# 解像度を設定
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 幅を1920ピクセルに設定
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1680) # 高さを1680ピクセルに設定

origins = []
tags=[]
joint_name_values =["PS-10SH","PS-18SU","PS-24SU","PS-33SU","TH-10","TH-18","TH-24","TH-33"]
joint_position = ["FRONT","BACK","SIDE"]
for obj_name in joint_name_values:
    for pattern in joint_position:
        output_dir = os.path.join(r'T:\Goro\ComputerVision\joints_data\generate', obj_name, pattern)
        file_name = "000.png" 
        file_path = os.path.join(output_dir, file_name)
        origin = cv2.imread(file_path)
        tag = obj_name + "@" + pattern
        origins.append(origin)
        tags.append(tag)


while True:
    ret, frame = capture.read()
    if ret:
        transform.main(frame,origins,tags)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
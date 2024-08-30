import cv2

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print('カメラを認識できません')
    exit()




width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

print('framesize = '+str(width)+' x '+str(height))
print('fps = '+str(fps))
print('frame count = '+str(frame_count))
import transform
import cv2



def getVision():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラからの映像を取得できませんでした。")
            break
        origins = []
        tags = []
        for i in range(1, 4):
            origin = cv2.imread(f"origin{i}.png")
            origins.append(origin)
            with open(f"tag{i}.txt") as f:
                tags.append(f.read())
        transform.main(frame, origins, tags)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif cv2.waitKey(1) & 0xFF == ord("s"):
            return frame
        
    cap.release()
    cv2.destroyAllWindows()
    return None





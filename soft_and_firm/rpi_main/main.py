from lib.vision.img_process import ImgProcess
import cv2
import numpy as np

improc = ImgProcess()
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
pucks_info = improc.detect_pucks(frame,np.array([100, 100, 100]), np.array([130, 255, 255]))
result_frame = frame.copy()
for puck in pucks_info:
    x1,y1,x2,y2,width,height,sim = puck["x1"],puck["y1"],puck["x2"],puck["y2"],puck["width"],puck["height"], puck["sim"]
    cv2.rectangle(result_frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(result_frame,str(sim),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

cv2.imshow("RESULT",result_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import json
import time

contours_for_compare = []

def load_contours(path: str):
    global contours_for_compare
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    contours_for_compare = []
    for template_contours in json_data:
        template_list = []
        for contour_data in template_contours:
            contour_array = np.array(contour_data, dtype=np.int32).reshape(-1, 1, 2)
            template_list.append(contour_array)
        contours_for_compare.append(template_list)

def check_contour(test_contour):
    best_match = float("inf")
    for template_contours in contours_for_compare:
        for template_contour in template_contours:
            similarity = cv2.matchShapes(test_contour, template_contour, 1, 0.0)
            if similarity < best_match:
                best_match = similarity
    return best_match

load_contours("../contours/contours.json")

def detect_pucks(frame,similarity_threshold,lower_color_threshold,upper_color_threshold,width_threshold=40):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color_threshold, upper_color_threshold)
    denoised_mask = cv2.medianBlur(mask, 5)
    contours, _  = cv2.findContours(denoised_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    detected_pucks = []

    for contour in contours:
        sim = check_contour(contour)
        
        if(sim<similarity_threshold):
            x,y,w,h = cv2.boundingRect(contour) 
            if(w>width_threshold):
                puck_info = {
                    "x1": int(x),
                    "y1": int(y),
                    "x2": int(x+w),
                    "y2": int(y+h),
                    "width": int(w),
                    "height": int(h),
                    "sim": sim
                }
                detected_pucks.append(puck_info)

    return detected_pucks

frame = cv2.imread("test_img.png")

start_time = time.time()

pucks_info = detect_pucks(frame,0.05,np.array([100, 100, 100]), np.array([130, 255, 255]))

end_time = time.time()
process_time = end_time-start_time
print(f"IMAGE PROCESSED IN {process_time}")


result_frame = frame.copy()
for puck in pucks_info:
    x1,y1,x2,y2,width,height,sim = puck["x1"],puck["y1"],puck["x2"],puck["y2"],puck["width"],puck["height"], puck["sim"]
    cv2.rectangle(result_frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(result_frame,str(sim),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

cv2.namedWindow("RESULT",cv2.WINDOW_NORMAL)
cv2.imshow("RESULT",result_frame)


cv2.waitKey(0)
cv2.destroyAllWindows()
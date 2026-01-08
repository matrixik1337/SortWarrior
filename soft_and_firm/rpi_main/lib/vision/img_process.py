# Lib for image processing

import cv2
import numpy as np
import json

# ALL COLORS is 0
# RED color is 1
# BLUE color is 2
# GREEN color is 3

red_lower = np.array([0,50,50])
red_upper = np.array([20,255,255])
blue_lower = np.array([110,50,50])
blue_upper = np.array([130,255,255])

red_ui_color = (127,127,255)
blue_ui_color = (255,127,127)

class ImgProcess:
    __contours_for_compare = []
      
    def __init__(self,path_to_contours_json="lib/vision/contours/contours.json"): # Loading templates from json
      with open(path_to_contours_json, 'r', encoding='utf-8') as f:
          json_data = json.load(f)
      
      for contours in json_data:
          list = []
          for contour_data in contours:
              contour_array = np.array(contour_data, dtype=np.int32).reshape(-1, 1, 2)
              list.append(contour_array)
          self.__contours_for_compare.append(list)      

    def __check_contour(self,contour): # Comparing contours with templates
      best_match = float("inf")
      for template_contours in self.__contours_for_compare:
          for template_contour in template_contours:
              sim= cv2.matchShapes(contour, template_contour, 1, 0.0)
              if sim < best_match:
                  best_match = sim
      return best_match


    def detect_pucks(self,frame,lower_color_threshold,upper_color_threshold,width_threshold=40,similarity_threshold=0.05): # Detect pucks by color and contour
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_color_threshold, upper_color_threshold)
        denoised_mask = cv2.medianBlur(mask, 5)
        contours, _  = cv2.findContours(denoised_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        detected_pucks = []

        for contour in contours:
            sim = self.__check_contour(contour)
            
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
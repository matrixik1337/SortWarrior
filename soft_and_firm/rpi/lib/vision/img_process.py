import cv2
import numpy as np

#RED is 0
#BLUE is 1

class ImgProcess:
    def get_pucks_count(self,frame,color,startX,startY,endX,endY):
      result_frame = frame.copy()
      cv2.rectangle(result_frame,(int(startX),int(startY)),(int(endX), int(endY)), (50,50,255),4)
    
      count = 0
      if color==0:
        lower = np.array([0,150,150])
        upper = np.array([0,255,255])
      elif color==1:
        lower = np.array([120,150,150])
        upper = np.array([120,255,255])

      hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
      mask = cv2.inRange(hsv_frame,lower,upper)
      contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      objData = []

      for _, c in enumerate(contours):
        boundRect = cv2.boundingRect(c)
        objData.append((count, boundRect))


        rectX = int(boundRect[0])
        rectY = int(boundRect[1])
        rectWidth = int(boundRect[2])
        rectHeight = int(boundRect[3])
        if rectX>startX and rectX+rectWidth<endX and rectY>startY and rectY+rectHeight<endY:
          count += 1 
          cv2.putText(result_frame,str("X:")+str((rectX+rectWidth)/2)+str(";")+str("Y:")+str((rectY+rectHeight)/2),(rectX,rectY-2),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
          cv2.rectangle(result_frame,(rectX, rectY), (rectX + rectWidth, rectY + rectHeight), (0,255,0), 2)

      cv2.putText(result_frame,str("PUCKS: ")+str(count),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)

      return count, result_frame

    def track_nearest_puck(self,frame,color):
      result_frame = frame.copy()
    
      count = 0
      if color==0:
        lower = np.array([250,150,150])
        upper = np.array([5,255,255])
      elif color==1:
        lower = np.array([115,150,150])
        upper = np.array([125,255,255])

      hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
      mask = cv2.inRange(hsv_frame,lower,upper)
      contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      objData = []

      x=0
      y=0
      width = 0
      height = 0

      for _, c in enumerate(contours):
        boundRect = cv2.boundingRect(c)
        objData.append((count, boundRect))


        rectX = int(boundRect[0])
        rectY = int(boundRect[1])
        rectWidth = int(boundRect[2])
        rectHeight = int(boundRect[3])
        if rectWidth>width:
          x=rectX
          y=rectY
          width=rectWidth
          height=rectHeight
          
      cv2.putText(result_frame,str("X:")+str((x+width)/2)+str(";")+str("Y:")+str((y+height)/2),(x,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)    
      cv2.rectangle(result_frame,(x, y), (x + width, y + height), (0,255,0), 2)

      cv2.drawMarker(result_frame,(int((x+width)/2),int((y+height)/2)),(255,255,255),cv2.MARKER_CROSS,1,1)

      return (x+width)/2, (y+height)/2, result_frame
    
    def track_pucks(self,frame,)
import cv2
import numpy as np

# ALL COLORS is 0
# RED color is 1
# BLUE color is 2

red_lower = np.array([0,50,50])
red_upper = np.array([20,255,255])
blue_lower = np.array([110,50,50])
blue_upper = np.array([130,255,255])

red_ui_color = (127,127,255)
blue_ui_color = (255,127,127)


class ImgProcess:
    def get_pucks_count(self,frame,color,startX,startY,endX,endY): # Gettings count of pucks on frame in predefined area
      result_frame = frame.copy()
      cv2.rectangle(result_frame,(int(startX),int(startY)),(int(endX), int(endY)), (50,50,255),4)
    
      count = 0

      # Range depended on color

      if color == 0:
        hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        objData = []

        lower = red_lower
        upper = red_upper
        mask = cv2.inRange(hsv_frame,lower,upper)
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for _, c in enumerate(contours): # Count and mark all pucks
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
            cv2.drawMarker(result_frame,(int(rectX+(rectWidth/2)),int(rectY+(rectHeight/2))),red_ui_color,cv2.MARKER_DIAMOND,10,3)

        lower = blue_lower
        upper = blue_upper
        mask = cv2.inRange(hsv_frame,lower,upper)
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for _, c in enumerate(contours): # Count and mark all pucks
          boundRect = cv2.boundingRect(c)
          objData.append((count, boundRect))


          rectX = int(boundRect[0])
          rectY = int(boundRect[1])
          rectWidth = int(boundRect[2])
          rectHeight = int(boundRect[3])
          if rectX>startX and rectX+rectWidth<endX and rectY>startY and rectY+rectHeight<endY:
            count += 1 
            cv2.putText(result_frame,str("X:")+str((rectX+rectWidth)/2)+str(";")+str("Y:")+str((rectY+rectHeight)/2),(rectX,rectY-2),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
            cv2.rectangle(result_frame,(rectX, rectY), (rectX + rectWidth, rectY + rectHeight),(0,255,0), 2)
            cv2.drawMarker(result_frame,(int(rectX+(rectWidth/2)),int(rectY+(rectHeight/2))),blue_ui_color,cv2.MARKER_DIAMOND,10,3)

      else:
        if color==1:
          lower = red_lower
          upper = red_upper
          ui_color = red_ui_color
        elif color==2:
          lower = blue_lower
          upper = blue_upper
          ui_color = blue_ui_color

        hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame,lower,upper)
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objData = []

        for _, c in enumerate(contours): # Count and mark all pucks
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
            cv2.drawMarker(result_frame,(int(rectX+(rectWidth/2)),int(rectY+(rectHeight/2))),ui_color,cv2.MARKER_DIAMOND,10,3)

      return count, result_frame

    def track_nearest_puck(self,frame,color,startX,startY,endX,endY): # Getting coordinates of nearest puck in predefined area
      result_frame = frame.copy()
      cv2.rectangle(result_frame,(int(startX),int(startY)),(int(endX), int(endY)), (50,50,255),4)


      x=0
      y=0
      width = 0
      height = 0

      if color==0:
        count=0

        hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        lower = red_lower
        upper = red_upper
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
            if rectWidth>width:
              x=rectX
              y=rectY
              width=rectWidth
              height=rectHeight


        lower = blue_lower
        upper = blue_upper
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
            if rectWidth>width:
              x=rectX
              y=rectY
              width=rectWidth
              height=rectHeight

      else:

        count=0
        if color==1:
          lower = red_lower
          upper = red_upper
        elif color==2:
          lower = blue_lower
          upper = blue_upper

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
          if rectX>startX and rectX+rectWidth<endX and rectY>startY and rectY+rectHeight<endY:
            if rectWidth>width:
              x=rectX
              y=rectY
              width=rectWidth
              height=rectHeight
            
      cv2.putText(result_frame,str("X:")+str(x+(width/2))+str(";")+str("Y:")+str(y+(height/2)),(x,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)    
      cv2.rectangle(result_frame,(x, y), (x + width, y + height), (0,255,0), 2)

      cv2.drawMarker(result_frame,(int(x+(width/2)),int(y+(height/2))),(255,255,255),cv2.MARKER_TILTED_CROSS,10,5)

      return x+(width/2), y+(height/2), result_frame
  
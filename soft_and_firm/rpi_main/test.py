# Test program

from lib.vision.img_process import ImgProcess
import cv2

improc = ImgProcess()

frame = cv2.imread("images_for_test/img1.png")


pucks_count, result_frame = improc.get_pucks_count(frame,0,10,10,920,650)
cv2.imshow("result_frame",result_frame)
print(f"Pucks count: {pucks_count}")
cv2.waitKey(1000)

red_pucks_count, result_frame = improc.get_pucks_count(frame,1,10,10,920,650)
cv2.imshow("result_frame",result_frame)
print(f"Red pucks count: {red_pucks_count}")
cv2.waitKey(1000)

blue_pucks_count, result_frame = improc.get_pucks_count(frame,2,10,10,920,650)
cv2.imshow("result_frame",result_frame)
print(f"Blue pucks count: {blue_pucks_count}")
cv2.waitKey(1000)

red_x,red_y, result_frame = improc.track_nearest_puck(frame,1,10,10,920,650)
cv2.imshow("result_frame",result_frame)
print(f"Coordinates of nearest red puck: x={red_x};y={red_y}")
cv2.waitKey(1000)

blue_x,blue_y, result_frame = improc.track_nearest_puck(frame,2,10,10,920,650)
cv2.imshow("result_frame",result_frame)
print(f"Coordinates of nearest blue puck: x={blue_x};y={blue_y}")
cv2.waitKey(1000)

x,y, result_frame = improc.track_nearest_puck(frame,0,10,10,920,650)
cv2.imshow("result_frame",result_frame)
print(f"Coordinates of nearest puck: x={x};y={y}")
cv2.waitKey(1000)


cv2.destroyAllWindows()
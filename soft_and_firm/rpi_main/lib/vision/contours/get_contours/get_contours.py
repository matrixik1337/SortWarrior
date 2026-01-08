import cv2
import numpy as np
import json

contours = []

for i in range(5):
    template_image = cv2.imread(f"puck{i+1}.png")
    if template_image is None:
        print(f"Cannot load \"puck{i+1}.png\"")
        continue

    template_gray_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    _, template_mask = cv2.threshold(template_gray_image, 150, 255, cv2.THRESH_BINARY_INV)
    
    contour_output = cv2.findContours(template_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    template_contour = contour_output[0] if len(contour_output) == 2 else contour_output[1]
    
    contours.append(template_contour)

    result = cv2.cvtColor(template_gray_image, cv2.COLOR_GRAY2BGR)
    result = cv2.drawContours(result, template_contour, -1, (0, 255, 0), 4)
    cv2.imshow(f"CONTOUR{i+1}", result)

print("Press \"q\" to continue")
while cv2.waitKey(10) != ord("q"):
    pass
cv2.destroyAllWindows()

if input("Write new contours? (y/n): ") == "y":
    json_data = []
    for contour_list in contours:
        json_data.append([c.tolist() for c in contour_list])

    with open("../contours.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    print("Successfilly writed to json")
    
else:
    print("Aborted.")
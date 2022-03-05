import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

def multi_temp(img_path='Assets/img_2.jpg',template_path='Assets/logo_1.png' , thresh=0.2):
    # Reading the image and template
    img = cv2.imread(img_path)
    temp = cv2.imread(template_path)
    tW,tH = temp.shape[:2]

    # Converting them to grayscale
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    temp_gray = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)

    # Passing the image to matchTemplate method 
    detect = cv2.matchTemplate(image=img_gray, templ=temp_gray, method=cv2.TM_CCOEFF_NORMED)

    (yCoords, xCoords) = np.where(detect >= thresh)
   
# initialize our list of rectangles
    rects = []
    # loop over the starting (x, y)-coordinates again
    for (x, y) in zip(xCoords, yCoords):
        # update our list of rectangles
        rects.append((x, y, x + tW, y + tH))

    # apply non-maxima suppression to the rectangles
    pick = non_max_suppression(np.array(rects))
    print("[INFO] {} matched locations *after* NMS".format(len(pick)))
    
    # loop over the final bounding boxes
    for (startX, startY, endX, endY) in pick:
        # draw the bounding box on the image
        cv2.rectangle(img, (startX, startY), (endX, endY),
            (255, 0, 0), 3)

    
    # show our output image *before* applying non-maxima suppression
    cv2.imshow("Template" ,temp)
    cv2.imshow("After NMS", img)
    cv2.waitKey(0)


#running the program: 
if (__name__) == '__main__':
    multi_temp()
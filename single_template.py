import cv2

def single_temp(img_path='Assets/img_1.jpg',template_path='Assets/logo_1.png'):
    # Reading the image and template
    img = cv2.imread(img_path)
    temp = cv2.imread(template_path)


    # Converting them to grayscale
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    temp_gray = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)

    # Passing the image to matchTemplate method 
    detect = cv2.matchTemplate(image=img_gray, templ=temp_gray, method=cv2.TM_CCOEFF_NORMED)
    
    # locating the most likely part of the image
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(detect) # we only the co-ordinates of the maximum -> maxLoc

    # Creating the coordinates for detection
    (startX, startY) = maxLoc
    endX = startX + temp.shape[1] 
    endY = startY + temp.shape[0]

    # Show the detection on the image
    cv2.rectangle(img,(startX,startY),(endX,endY),(255,0,0),2)

    cv2.imshow('Template',temp)
    cv2.imshow('Image',img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

# running the program

if (__name__) == '__main__':
    single_temp()
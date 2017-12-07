import cv2
import numpy as np

ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),5,(255,0,0),0)
        ix,iy = x,y

# Create a black image, a window and bind the function to window

frames = 75
cv2.namedWindow('image')

for i in range(frames):
    img = cv2.imread('human_pics/%02d.jpg' %(i))
    
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            print (ix,iy)
            break
cv2.destroyAllWindows()

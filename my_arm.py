from arm_env import ArmEnv
import pyglet
from pyglet.window import key
import random
import cv2
import numpy as np

MODE = ['easy', 'hard']
n_model = 0

env = ArmEnv(mode=MODE[n_model])
env.set_fps(30)


env.render()
env.step([3,3])
env.render()
while True:
    env.step([0,0])
    env.render()
    #for j in range(100000000):
    #    continue




    #cv2.imwrite('loop'+str(t)+'.jpg',im)
    
    #v1 = random.randint(1, 10)
    #v2 = random.randint(1, 10)
    

    #env.step2([100,100])
    #env.render()

    #env.step2([100,50])
    #im = env.getImage()
    '''
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 200,255,cv2.THRESH_BINARY_INV)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    im3 = np.zeros(im.shape, np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400*350:
            continue
        else:
            cv2.drawContours(im3, [cnt], -1, (255,255,255), -1)
        #cv2.drawContours(im, [cnt], -1, (255,0,0), -1)'''

    #cv2.imshow('img',im)
    #cv2.imwrite('loop'+str(t)+'.jpg',im)
    #cv2.imshow('thresh',thresh)

    #cv2.waitKey(100)
  

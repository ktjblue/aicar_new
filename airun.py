import xhat as hw
import time
import cv2
import config as cfg
#import opidistance3 as dc
import tensorflow as tf
import scipy.misc
import numpy as np
import model

import os
import sys
import signal
import csv

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "save/model.ckpt")

    start_flag = False

    #testing speed variation
    speed_change_flag = False

    if speed_change_flag:
        cfg.maxturn_speed = cfg.ai_maxturn_speed
        cfg.minturn_speed = cfg.ai_minturn_speed
        cfg.normal_speed_left = cfg.ai_normal_speed_left
        cfg.normal_speed_right = cfg.ai_normal_speed_right
    
    c = cv2.VideoCapture(0)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    #c.set(cv2.CAP_PROP_FPS, 15)

    while(True):
        _, full_image = c.read()
        # full_image = cv2.resize(full_image, (320,240))
        image = scipy.misc.imresize(full_image[cfg.modelheight:], [66, 200]) / 255.0
        image1 = scipy.misc.imresize(full_image[cfg.modelheight:], [66*2, 200*2])

        #cv2.imshow('original',full_image)
        #cv2.imshow("view of AI", cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
        cv2.imshow("view of AI", image1)


        wheel = model.y.eval(session=sess,feed_dict={model.x: [image], model.keep_prob: 1.0})
        cfg.wheel = np.argmax(wheel, axis=1)
        #print('wheel value:', cfg.wheel, wheel)
        print('wheel value:', cfg.wheel, model.softmax(wheel))

    
        k = cv2.waitKey(5)
        if k == ord('q'):  #'q' key to stop program
            break

        """ Toggle Start/Stop motor movement """
        if k == ord('a'): 
            if start_flag == False: 
                start_flag = True
            else:
                start_flag = False
            print('start flag:',start_flag)
   
        #to avoid collision when ultrasonic sensor is available
        length = 30 #dc.get_distance()
        if  5 < length and length < 15 and start_flag:
            hw.motor_one_speed(0)
            hw.motor_two_speed(0)
            print('Stop to avoid collision')
            time.sleep(0.5)
            continue
        
        if start_flag:
            if cfg.wheel == 0:
                hw.motor_two_speed(0)
                hw.motor_one_speed(0)

            if cfg.wheel == 1:   #left turn
                hw.motor_two_speed(cfg.minturn_speed)
                hw.motor_one_speed(cfg.maxturn_speed)
            

            if cfg.wheel == 2:
                hw.motor_two_speed(cfg.normal_speed_left)
                hw.motor_one_speed(cfg.normal_speed_right)

            if cfg.wheel == 3:   #right turn
                hw.motor_two_speed(cfg.maxturn_speed)
                hw.motor_one_speed(cfg.minturn_speed)
        
        else:
            hw.motor_one_speed(0)
            hw.motor_two_speed(0)
            cfg.wheel = 0

        
hw.motor_clean()
cv2.destroyAllWindows()

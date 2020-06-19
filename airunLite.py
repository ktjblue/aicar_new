#import xhat as hw
import xhat_temp as hw
import time
import cv2
import config as cfg
import tensorflow as tf

import scipy.misc
import numpy as np
#import model

import os
import sys
import signal
import csv
import socket
import fcntl
import time

#import keras
#from keras import backend as K

if __name__ == '__main__':
    
    s = socket.socket()
    port = int(str(sys.argv[1]))
    
    if len(sys.argv) < 2:
        print('please enter proper port number..')
        sys.exit()
        
    s.connect(('192.168.1.193',port))
    print('Connected!')
    s.setblocking(0)
        
            
    #cmd = s.recv(1024)
    #print(str(cmd))
    #sys.exit()
    #fcntl.fcntl(s, fcntl.F_SETFL, os.O_NONBLOCK)
        
    #K.set_learning_phase(0)
    #keras_model = tf.keras.models.load_model('save/my_model.h5')
    interpreter = tf.lite.Interpreter("save/my_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()
    print(output_details)
     
    start_flag = False
    slow_flag = False

    c = cv2.VideoCapture(0)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    #c.set(cv2.CAP_PROP_FPS, 15)

    while(True): 
        
        _,full_image = c.read()

        image = full_image[cfg.modelheight:]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, dsize=(cfg.final_width, cfg.final_height)) / 255.0
        image = np.array(image)
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)

        image_view = cv2.resize(full_image[cfg.modelheight:], (cfg.final_width*2,cfg.final_height*2))
        cv2.imshow("View of AI", image_view)

        #wheel = keras_model.predict(image, batch_size=1)
        interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
        interpreter.invoke()
        wheel = interpreter.get_tensor(output_details[0]['index'])
        cfg.wheel = np.argmax(wheel, axis=1)
        #print('wheel value:', cfg.wheel)
        print('wheel value:', cfg.wheel, wheel)
        #if cfg.wheel==cfg.STOP:
        #    input('enter')
            
        #print('wheel value:', cfg.wheel, model.softmax(wheel))

        try:
            cmd = s.recv(1024)
            print(str(cmd))
            
            if cmd == b'start the car':
                start_flag = True
            elif cmd == b'stop the car':
                start_flag = False
            elif cmd == b'accident':
                slow_flag = True
            elif cmd == b'clear':
                slow_flag = False
            
                
        except:
            print("Waiting for the signal...")
  
    
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
            if cfg.wheel == cfg.STOP:
                hw.motor_one_speed(cfg.normal_speed_right==0)
                hw.motor_two_speed(cfg.normal_speed_left==0)
            elif cfg.wheel == cfg.LEFT:   #left turn
                hw.motor_one_speed(cfg.maxturn_speed)
                hw.motor_two_speed(cfg.minturn_speed)
            elif cfg.wheel == cfg.UP:
                #hw.motor_one_speed(cfg.normal_speed_right)
                #hw.motor_two_speed(cfg.normal_speed_left)
                if slow_flag == True:
                    hw.motor_one_speed(int(cfg.normal_speed_right/2))
                    hw.motor_two_speed(int(cfg.normal_speed_left/2))
                else:
                    hw.motor_one_speed(cfg.normal_speed_right)
                    hw.motor_two_speed(cfg.normal_speed_left)
            elif cfg.wheel == cfg.RIGHT:   #right turn
                hw.motor_one_speed(cfg.minturn_speed)
                hw.motor_two_speed(cfg.maxturn_speed)
            elif cfg.wheel == cfg.UP_LEFT:   
                hw.motor_one_speed(cfg.normal_speed_right)
                hw.motor_two_speed(round(cfg.normal_speed_left/cfg.slow_turn_factor))
            elif cfg.wheel == cfg.UP_RIGHT:   #right turn
                hw.motor_one_speed(round(cfg.normal_speed_right/cfg.slow_turn_factor))
                hw.motor_two_speed(cfg.normal_speed_left)
            else:
                assert False
        
        else:
            hw.motor_one_speed(0)
            hw.motor_two_speed(0)
            cfg.wheel = cfg.STOP

        
hw.motor_clean()
cv2.destroyAllWindows()
s.close()

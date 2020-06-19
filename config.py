# -*- coding: utf-8 -*-


# Configuration
width = 320  # Video width requested from camera
height = 240  # Video height requested from camera

#final_width = 200
final_width = 100
final_height = 66

wheel = 0  #0:stop, 1:left, 2:strait, 3:right

recording = False

cnt = 0
#outputDir = '/home/orangepi/autonomousCar/lesson4/data/'
outputDir = '/home/orangepi/Desktop/autoCar/data/'
currentDir = 'training'
file = ""
f = ''
fwriter = ''

Voicecontrol = False

AIcontrol = False
modelheight = -160 ###-130 ###-150 #-115 #-130 #-150 #-250 #-200

# training speed setting

maxturn_speed = 100
minturn_speed = 3  ###20  ###15
normal_speed_left = 60
normal_speed_right = 60
wheel_alignment_left = 0
wheel_alignment_right = 0
slow_turn_factor = 5
stop_speed = 0


# testing speed setting(
"""
ai_maxturn_speed = 100
ai_minturn_speed = 20
ai_normal_speed_left = 100
ai_normal_speed_right = 100
"""
ai_maxturn_speed = 1
ai_minturn_speed = 1
ai_normal_speed_left = 1
ai_normal_speed_right = 1

# taewoon added
sleep_time_second = 0.100

IMG_WIDTH = 320
IMG_HEIGHT = 240

KEY_s = 115
KEY_r = 114
KEY_LEFT_ARROW = 81
KEY_RIGHT_ARROW = 83
KEY_UP_ARROW = 82
KEY_j = 106  # left
KEY_l = 108  # right
KEY_i = 105  # up
KEY_u = 117  # up left
KEY_o = 111  # up right

STOP = 0
LEFT = 1
UP = 2
RIGHT = 3
#NUM_KEYS = 4
UP_LEFT = 4
UP_RIGHT = 5
NUM_KEYS = 6

# EOF
import time
import cv2
import joblib
import numpy as np
from PIL import ImageGrab
from pynput import keyboard
from pynput.keyboard import Key

time.sleep(3)
# 0、创建键盘
kb = keyboard.Controller()
# 1、加载模型
left_or_right = joblib.load('left_or_right.m')
jump_time = joblib.load('long.m')
while True:
    # 2、准备数据
    ImageGrab.grab().resize((960, 540)).save('current.jpg')  # 保存当前屏幕截屏
    x = cv2.imread('current.jpg', 0).reshape(-1)
    x = [x]
    pred1 = left_or_right.predict(x)
    pred2 = jump_time.predict(x)
    print(pred1,pred2)

    if pred1[0] == 0 and pred2[0]==0:        #左走
        kb.press(Key.left)
        time.sleep(0.2)
        kb.release(Key.left)        
    if pred1[0] == 1 and pred2[0]==0:        #右走
        kb.press(Key.right)
        time.sleep(0.2)
        kb.release(Key.right)

    if pred1[0] == 0 and pred2[0]==1:        #左小跳
        kb.press(Key.left)
        kb.press(Key.space)
        time.sleep(0.2)
        kb.release(Key.left)
        kb.release(Key.space)
    if pred1[0] == 1 and pred2[0]==1:        #右小跳
        kb.press(Key.right)
        kb.press(Key.space)
        time.sleep(0.2)
        kb.release(Key.right)
        kb.release(Key.space)

    if pred1[0] == 0 and pred2[0]==2:        #左中跳
        kb.press(Key.left)
        kb.press(Key.space)
        time.sleep(0.6)
        kb.release(Key.left)
        kb.release(Key.space)
    if pred1[0] == 1 and pred2[0]==2:        #右中跳
        kb.press(Key.right)
        kb.press(Key.space)
        time.sleep(0.6)
        kb.release(Key.right)
        kb.release(Key.space)

    if pred1[0] == 0 and pred2[0]==3:        #左大跳
        kb.press(Key.left)
        kb.press(Key.space)
        time.sleep(1.2)
        kb.release(Key.left)
        kb.release(Key.space)
    if pred1[0] == 1 and pred2[0]==3:        #右大跳
        kb.press(Key.right)
        kb.press(Key.space)
        time.sleep(1.2)
        kb.release(Key.right)
        kb.release(Key.space)
    time.sleep(2)  
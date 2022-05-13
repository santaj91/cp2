import cv2
import tensorflow as tf
import numpy as np
import glob
import dlib
from keras.models import load_model

from function import crop , load_imgs


##----------------------------------------------- load the model-----------------------------------------------------------------##
model = load_model('./model/facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)
model.load_weights('./model/facenet_keras_weights.h5')


size = (448,448) # 나중에 수정  
input_size= (160,160)

##---------------------------------------------Load images ( n <=5 )--------------------------------------------------------------##







##--------------------------------------------------------------------------------------------------------------------------------##


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while cap.isOpened(): # 캡처 객체 초기화 확인 cap 객체가 지정한 파일이 정상적으로 초기화 -> True
    ret, img_ori = cap.read() # 다음 프레임 읽기 / 프레임 잘 읽었으면 ret = True , img_ori는 프레임 이미지

    if not ret:
        break

    img = img_ori.copy
    img = cv2.resize(img,size)
    video_crops = crop.crop(img,input_size)# 리스트 형태의 nomalizing 된 cropped face image
    
    for video_face in video_crops: # 여러개일 가능성도 있기 떄문에 
        prediction_video=model.predict(video_face) # video에 나온 얼굴







    if cv2.waitKey(1) == ord('q'): # 1ms 의 지연을 주면서 화면에 표시, q 누르면 break
        break


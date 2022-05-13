import cv2
import tensorflow as tf
import numpy as np
import glob
import dlib
from keras.models import load_model

from function import crop , load_imgs



img_path="./image"
size = (448,448) # 나중에 수정  
input_size= (160,160)# 모델에 삽입되는 사이즈

##----------------------------------------------- load the model-----------------------------------------------------------------##
model = load_model('./model/facenet_keras.h5')
# summarize input and output shape
model.load_weights('./model/facenet_keras_weights.h5')

##---------------------------------------------Load images ( n <=5 )--------------------------------------------------------------##

# 저장된 이미지 로드
images = load_imgs.load_imgs(img_path,size) # 리스트 반환

for image in images:
    faces=crop.crop(image,input_size) # 얼굴 이미지 (얼굴이 여러개 잡힐 때) 
    for face in faces:
        predictions=[]
        prediction=model.predict(face)
        predictions.append(prediction)



##-------------------------------------------video capture and comparing ---------------------------------------------------------##


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while cap.isOpened(): # 캡처 객체 초기화 확인 cap 객체가 지정한 파일이 정상적으로 초기화 -> True
    ret, img_ori = cap.read() # 다음 프레임 읽기 / 프레임 잘 읽었으면 ret = True , img_ori는 프레임 이미지

    if not ret:
        break

    img = img_ori.copy()
    img = cv2.resize(img,dsize=size)
    
    video_crops = crop.crop(img,input_size)# 리스트 형태의 nomalizing 된 cropped face image
    
    for video_face in video_crops: # 실시간 얼굴이 여러개일 가능성도 있기 떄문에
        video_face=video_face.copy().reshape((-1, input_size[1], input_size[0], 3)).astype(np.float32) / 255
        prediction_video=model.predict(video_face) # video에 나온 얼굴 vector

        for pred_img in predictions:# 실시간 사진과 이미지 여러장 만큼 대조
            euclidean_distance = np.linalg.norm(prediction_video-pred_img)
            if euclidean_distance <= 1.1:
                break
            else:
                print(euclidean_distance)
                continue



    cv2.imshow('',img)
    if cv2.waitKey(1) == ord('q'): # 1ms 의 지연을 주면서 화면에 표시, q 누르면 break
        break

print('pass')


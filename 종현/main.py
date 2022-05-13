import cv2
import tensorflow as tf
import numpy as np
import glob
import dlib
from keras.models import load_model


def load_imgs(img_path,size):
    result=[]
    imgs_path_list= [x for x in glob.glob(img_path + '/*.jpg')]

    for img_path in imgs_path_list:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img,size)
        result.append(img)
    return result # 리스트 형태로 여러장 이미지 반환

def crop(image,input_size):# 이미지 하나, 모델에 들어갈 사이즈
    imgs=[]
    detector = dlib.get_frontal_face_detector()
    faces = detector(image)
    for face in faces:
        face_img=image[face.top():face.bottom(), face.left():face.right()]
        face_img=cv2.resize(face_img,dsize=input_size)
        face_input=face_img.copy().reshape((input_size[1], input_size[0], 3)).astype(np.float32)/255.
        imgs.append(face_input)
    return imgs# 얼굴 여러개


# load the model
model = load_model('./model/facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)
model.load_weights('./model/facenet_keras_weights.h5')






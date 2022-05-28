import dlib
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import Image


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

def mtcnn(image, input_size):
    imgs = []
    detector = MTCNN()
    detector_result = detector.detect_faces(image)
    x1, y1, width, height = detector_result[0]['box']
    x2, y2 = x1 + width, y1 + height
    pixels = np.array(image)
    face = pixels[y1:y2, x1:x2]
    img = Image.fromarray(face) # 배열 객체를 입력으로 받아 배열 객체에서 만든 이미지 객체를 반환
    img = img.resize(input_size)
    face_array = np.asarray(img)
    imgs.append(face_array)
    return imgs
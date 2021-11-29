from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# load model
model = load_model('C:/Users/TEJPAL KUMAWAT/Downloads/gender_detection.model')
age_net = cv2.dnn.readNetFromCaffe('C:/Users/TEJPAL KUMAWAT/google drive api/deploy_age.prototxt.txt', 'C:/Users/TEJPAL KUMAWAT/google drive api/age_net.caffemodel')
fontFace = cv2.FONT_HERSHEY_SIMPLEX
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
classes = ['man', 'woman']

font = cv2.FONT_HERSHEY_SIMPLEX  # Type of font

frame=cv2.imread('C:/Users/TEJPAL KUMAWAT/google drive api/frame55.jpg')
path='C:/Users/TEJPAL KUMAWAT/google drive api/haarcascade_frontalface_default.xml'
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(path)
def detect(frame):
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if frame is ():
            return None
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cropped_img = frame[y:y+h, x:x+w]

        return cropped_img


cropped_image=detect(frame)


face_crop = cv2.resize(cropped_image, (96, 96))
face_crop = face_crop.astype("float") / 255.0
face_crop = img_to_array(face_crop)
face_crop = np.expand_dims(face_crop, axis=0)
conf = model.predict(face_crop)[0]
idx = np.argmax(conf)
label = classes[idx]

blob = cv2.dnn.blobFromImage(cropped_image, 1, (244, 244), MODEL_MEAN_VALUES, swapRB=True)
age_net.setInput(blob)
age_preds = age_net.forward()
age = age_list[age_preds[0].argmax()]


print(label)
print(age)




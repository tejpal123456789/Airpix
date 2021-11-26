from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# load model
model = load_model('C:/Users/TEJPAL KUMAWAT/Downloads/gender_detection.model')
#model1=load_model('C:/Users/TEJPAL KUMAWAT/google drive api/age_detect_cnn_model.h5')
#model=load_model('gender_detection.model')
model1=load_model('age_detect_cnn_model.h5')
model2=load_model('age_detection.h5')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
# open webcam
webcam = cv2.VideoCapture(0)

classes = ['man', 'woman']
#age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
font = cv2.FONT_HERSHEY_SIMPLEX  # Type of font
fontFace = cv2.FONT_HERSHEY_SIMPLEX
# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_detect = np.copy(frame[startY:endY, startX:endX])

        if (face_detect.shape[0]) < 10 or (face_detect.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_detect, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        conf = model.predict(face_crop)[0]
        
        #model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        face_roi = cv2.resize(face_detect, (200, 200))
        face_roi = face_roi.reshape(-1, 200, 200, 1)
        face_age = age_ranges[np.argmax(model1.predict(face_roi))]
       # face_age_pct = f"({round(np.max(model1.predict(face_roi)) * 100, 2)}%)"
        # get label with max accuracy
       # idx = np.argmax(conf)
      #  label = classes[idx]
       # blob = cv2.dnn.blobFromImage(face_crop, 1, (96,96), MODEL_MEAN_VALUES, swapRB=True)
       # age_net.setInput(blob)
       # age_preds = age_net.forward()
        #age = age_list[age_preds[0].argmax()]
        #overlay_text = "%s %s" % (label, age)
       #3 cv2.putText(frame, overlay_text, (startX, startY), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
       # cv2.imshow('frame', frame)
        # apply gender detection on face
        conf = model.predict(face_crop)[0]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10
        Z= Y = endY - 10 if endY - 10 > 10 else endY + 10
        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(frame,x,(endX, Z), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, color=(0, 255,0))

    # display output

    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()







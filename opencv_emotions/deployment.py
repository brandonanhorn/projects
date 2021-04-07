import numpy as np
import cv2

import tensorflow.keras
from tensorflow.keras.models import load_model

haar_classifier = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
emotion_model = load_model("model/callback_model.h5")

emotion_dim = (64,64)
emotion_dict = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}

color = (0, 250, 0,)
thickness = 3

vid_frames = []
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1,1)
    clone = frame.copy()
    bboxes = haar_classifier.detectMultiScale(clone)
    for i in bboxes:
        x, y, width, height = i[0], i[1], i[2], i[3]
        x2, y2 = x + width, y+ height
        emotion_roi = clone[y:y2, x:x2]
        emotion_roi = cv2.resize(emotion_roi, emotion_dim, interpolation = cv2.INTER_CUBIC)

        #preprocess emotion input
        emotion_roi = emotion_roi/255

        #resize emotion and mask to feed into nn
        emotion_roi = emotion_roi.reshape(1, emotion_roi.shape[0], emotion_roi.shape[1], emotion_roi.shape[2])

        #emotion predictions
        emotion_predict = emotion_model.predict(emotion_roi)[0]
        emotion_idx = np.argmax(emotion_predict)
        emotion_cat = emotion_dict[emotion_idx]
        emotion_conf = f'{round(np.max(emotion_predict)*100)}%'
        cv2.putText(clone, f'{emotion_cat}: {emotion_conf}', (x+80, y+50), cv2.FONT_HERSHEY_SIMPLEX, .80, color)
        cv2.rectangle(clone, (x,y), (x2,y2), 1)
        continue

    cv2.imshow('LIVE', clone)
    vid_frames.append(clone)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

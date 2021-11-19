import streamlit as st
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("data/mask_detector.model")

def app():
    st.title("Ứng dụng phát hiện không đeo khẩu trang")
    text_cam = st.text_input('Nhập địa chỉ camera IP của bạn. Nếu dùng webcam nhập 0 (hoặc 1,.. nếu có nhiều webcam)')
    faceCascade= cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    run = st.checkbox('Mở webcam/ camera IP')
    FRAME_WINDOW = st.image([])
    if text_cam.isnumeric():
        camera = cv2.VideoCapture(int(text_cam))
    else:
        camera = cv2.VideoCapture(str(text_cam))
    camera.set(3, 640)
    camera.set(4, 480)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray,1.4, 4, minSize=(30,30))
        for (x,y,w,h) in faces:
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            # ordering, resize it to 224x224, and preprocess it
            face = frame[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]

            label = "mask" if mask > withoutMask else "no_mask"
            color = (0, 255, 0) if label == "mask" else (255, 0, 0)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')

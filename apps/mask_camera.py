from streamlit_webrtc import webrtc_streamer
import av
import streamlit as st

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("data/mask_detector.model")
faceCascade= cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

def app():
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            flipped = img[::-1,:,:]

            return av.VideoFrame.from_ndarray(flipped, format="bgr24")


    webrtc_streamer(key="example", video_processor_factory=VideoProcessor)




    # class VideoProcessor:
    #     def recv(self, frame):
    #         img = frame.to_ndarray(format="bgr24")
            
    #         imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #         faces = faceCascade.detectMultiScale(imgGray,1.4, 4, minSize=(30,30))
    #         for (x,y,w,h) in faces:
    #             # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #             # ordering, resize it to 224x224, and preprocess it
    #             face = img[y:y+h, x:x+w]
    #             face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    #             face = cv2.resize(face, (224, 224))
    #             face = img_to_array(face)
    #             face = preprocess_input(face)
    #             face = np.expand_dims(face, axis=0)

    #             (mask, withoutMask) = model.predict(face)[0]

    #             label = "mask" if mask > withoutMask else "no_mask"
    #             color = (0, 255, 0) if label == "mask" else (255, 0, 0)

    #             label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

    #             cv2.putText(img, label, (x, y - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    #             cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    #         # flipped = img[::-1,:,:]

    #         return av.VideoFrame.from_ndarray(img, format="bgr24")

    # webrtc_streamer(key="example", video_processor_factory=VideoProcessor)



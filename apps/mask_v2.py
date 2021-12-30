from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import av
import streamlit as st
from mtcnn.mtcnn import MTCNN

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("data/mask_detector.model")


###########################
percent_plus = 0.015
threshold =0.95
###########################
def app():

    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            hight, width = img.shape[:2]
            plus_h = int(hight * percent_plus)
            plus_w = int(width * percent_plus)
            # img = cv2.resize(img, (300,300))
            # Change chanel color
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Model MTCNN
            detector = MTCNN()
            # Predict
            faces = detector.detect_faces(img_rgb)

            for face in faces:
                if face['confidence'] > threshold:
                    confident = face['confidence']
                    x,y,w,h = face['box']

                    if x >= plus_w:
                        x1 = x-plus_w
                    else:
                        x1 =x
                    
                    if y >= plus_h:
                        y1 = y-plus_h
                    else:
                        y1 =y

                    if x+ w+ plus_w <= width:
                        x2 = x+ w+ plus_w
                    else:
                        x2 = x+w
                    
                    if y+ h + plus_h <= hight:
                        y2 = y+ h + plus_h
                    else:
                        y2 = y+ h

                    face = img[y1:y2, x1:x2]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)

                    (mask, withoutMask) = model.predict(face)[0]

                    label = "mask" if mask > withoutMask else "no_mask"
                    color = (0, 255, 0) if label == "mask" else (0, 0, 255)

                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="example", video_processor_factory=VideoProcessor, \
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False})
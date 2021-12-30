from streamlit_webrtc import webrtc_streamer
import av
import cv2
from apps.yolo3 import *

##############################
frameWidth = 640
frameHeight = 480
image_h, image_w = 480, 640
input_w, input_h = 416, 416
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
class_threshold = 0.6
# You can choice label by yourself
# Example: labels = ["person", "car"]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
##############################

from google_drive_downloader import GoogleDriveDownloader as gdd
gdd.download_file_from_google_drive(file_id='1bhTkqX_I-JU7zGCi0owRTmfdw0QW-z15',
                                    dest_path='./yolov3.h5')
model = load_model('yolov3.h5')

# Model
# PyTorch Hub
def app():
    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            image = cv2.resize(img, (input_w, input_h))
            image = image.astype('float32')
            image /= 255.0
            # add a dimension so that we have one sample
            image = expand_dims(image, 0)
            yhat = model.predict(image)
            boxes = list()
            for i in range(len(yhat)):
            # decode the output of the network
                boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
            # correct the sizes of the bounding boxes for the shape of the image
            correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
            # suppress non-maximal boxes
            do_nms(boxes, 0.5)
            # get the details of the detected objects
            v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
            # summarize what we found
            for i in range(len(v_boxes)):
                print(v_labels[i], v_scores[i])
            for i in range(len(v_boxes)):
                box = v_boxes[i]
                # get coordinates
                y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
                # calculate width and height of the box
                width, height = x2 - x1, y2 - y1
                # create the shape
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # draw text and score in top left corner
                label = "%s (%.3f)" % (v_labels[i], v_scores[i])
                cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 3)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    webrtc_streamer(key="example", video_processor_factory=VideoProcessor, \
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False})


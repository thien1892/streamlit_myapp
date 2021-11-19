from streamlit_webrtc import webrtc_streamer
import av

def app():
    class VideoProcessor:
        def __init__(self):
            self.some_value = 0.5

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            
            self.do_something(img, self.some_value)  # `some_value` is used here
            

            return av.VideoFrame.from_ndarray(img, format="bgr24")


    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
    s_value = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

    # DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    if ctx.video_processor:
        ctx.video_processor.some_value = s_value
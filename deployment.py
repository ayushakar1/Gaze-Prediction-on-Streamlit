from streamlit_webrtc import webrtc_streamer
from demo2 import gazePrediction


webrtc_streamer(key="example", video_frame_callback=gazePrediction)


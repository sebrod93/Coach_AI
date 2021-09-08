import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import mediapipe as mp
import os
import tempfile
import requests


'''
# CoachAI
'''

st.markdown('''
Hello World !
''')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.landmarks = {}
        self.i = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(img)
            # self.landmarks[i] = results.
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        return img

webrtc_streamer(key="example", video_processor_factory=VideoTransformer)


video = st.file_uploader('video')
if video:
    url = 'http://127.0.0.1:8000/predict'
    data = {'video': video}
    x = requests.post(url, files= data)
    st.write(x.text)

st.video(video)
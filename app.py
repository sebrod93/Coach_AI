import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import mediapipe as mp
import os
import requests
import tempfile
from PIL import Image

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

stframe = st.empty()
video = st.file_uploader('video')
tfile = tempfile.NamedTemporaryFile(delete=False)
if video:
    url = 'https://coachai-ujeungn6tq-ew.a.run.app/predict'
    data = {'video': video}
    x = requests.post(url, files= data)
    st.write(x.text)

    #Load a video using cv2
    tfile.write(video.read())
    vid = cv2.VideoCapture(tfile.name)

    #Store the width and height for the current video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output.mp4', codec, fps_input, (width, height))

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        #Iterates through each frame in the current video
        while vid.isOpened():

            ret, frame = vid.read()
            if not ret:
                continue

            #Create an image of the current frmae
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            #Stores the landmarks of the current frame
            pose_landmarks = results.pose_landmarks

            mp_drawing.draw_landmarks(image=frame,
                                      landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.
                                      get_default_pose_landmarks_style())

            vid.write(frame)

    annotated_video = open(out,'rb')
    out_bytes = annotated_video.read()
    st.video(out_bytes)

    out.release()
    vid.release()

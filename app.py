import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import mediapipe as mp
import os
import requests
import tempfile
from PIL import Image
import io
from Coach_AI.data import json_to_df
import pandas as pd
from Coach_AI.utils import download_model

'''
# CoachAI
'''

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.landmarks = {}
        self.i = 0
        self.landmark_dict = {}
        self.data = []

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        frame_width = int(img.shape[1])
        frame_height = int(img.shape[0])
        
        j = 0
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(img)
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            pose_landmarks = results.pose_landmarks

            if pose_landmarks != None:
                for lmk in pose_landmarks.landmark:
                    self.landmarks[j] = [lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                    j += 1
            self.landmark_dict[self.i] = self.landmarks
            self.i+=1

            if len(self.landmark_dict) >= 40:
                self.data.append(json_to_df(self.landmark_dict)[0])
                print("--------self data [0]--------")
                print(self.data[0])

                cols = ['min_d0', 'max_d0', 'min_d1', 'max_d1', 'min_d2', 'max_d2', 'min_d3', 'max_d3', 'min_d4', 'max_d4',
                'min_d5', 'max_d5', 'min_d6', 'max_d6', 'min_d7', 'max_d7', 'min_d8', 'max_d8', 'min_d9', 'max_d9',
                'min_d10', 'max_d10', 'min_d11', 'max_d11', 'min_d12', 'max_d12', 'min_a0', 'max_a0', 'min_a1', 'max_a1',
                'min_a2', 'max_a2', 'min_a3', 'max_a3', 'min_a4', 'max_a4', 'min_a5', 'max_a5','mean_body_angle']

                X = pd.DataFrame(self.data)
                X.columns=cols

                model = download_model()
                results = model.predict(X)

                category = {0:'Push Up', 1:'Jumping Jack', 2:'Squat', 3:'Lunge', 4:'Pull Up'}
                print(results[0])
                print(category[results[0]])
            
        return img


webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

video = st.file_uploader('video')

if video:
    url = 'https://coachai-ujeungn6tq-ew.a.run.app/predict'
    data = {'video': video}
    x = requests.post(url, files= data)
    st.write(x.text)
    st.video(video)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())
    vf = cv2.VideoCapture(tfile.name)
    frame_width = int(vf.get(3))
    st.write(frame_width)
    frame_height = int(vf.get(4))
    st.write(frame_height)
    inflnm, inflext = video.name.split('.')
    exercise_name =  inflnm.split('_')[1]
    annotated_filename = f'{inflnm}_annotated.{inflext}'
    out_annotated = cv2.VideoWriter(annotated_filename, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        st.write(vf.isOpened())

        while vf.isOpened():
            ret, image = vf.read()
            if not ret:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            out_annotated.write(image)
        
        out_annotated.release()
        vf.release()

    st.video(annotated_filename)

    
from requests.models import Response
import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import mediapipe as mp
import os
import requests
import tempfile
from PIL import Image
import matplotlib.pyplot as plt
from Coach_AI.exercices import exercices
from Coach_AI.utils import make_plot
import numpy as np

emojis = {
    'cardio': '🫀',
    'pecs and arms': '💪🏻',
    'thighs': '🦵🏻',
    'glutes and calfs': '🍑',
    'back muscles': '🔙'
}

##Instantiate Mediapipe Objects
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

'''
# Coach AI 💪🏻 💪🏻 💪🏻
'''

#Sidebar Objects
st.sidebar.header('Upload your video :video_camera:')

video = st.sidebar.file_uploader('Video File',
                                 type=["mp4", "mov", 'avi', 'asf', 'm4v'])

if video:
    url = 'https://coachai-ujeungn6tq-ew.a.run.app/predict'
    data = {'video': video}

    response = requests.post(url, files=data).json()

    params_list = [np.array(element) for element in response['lists']]

    params_list.append(response['label'])

    st.write(f"Well done! You did {response['reps']} {response['prediction']}s! 🥇")

    target = exercices[response['prediction']]["Target_Muscle_Group"]

    st.write(
        f'You worked on your {target} {emojis[target]}!'
    )

    st.write(f'You are one step closer to your goals! 🏆')

    st.sidebar.video(video)

    st.sidebar.header('Good form example')

    st.sidebar.video(exercices[response['prediction']]["video_src"])

    ##Generate the annotated video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())
    vf = cv2.VideoCapture(tfile.name)
    frame_width = int(vf.get(3))
    frame_height = int(vf.get(4))
    fps = int(vf.get(cv2.CAP_PROP_FPS))
    inflnm, inflext = video.name.split('.')
    exercise_name = inflnm.split('_')[1]
    annotated_filename = f'{inflnm}_annotated.{inflext}'
    out_annotated = cv2.VideoWriter(annotated_filename,
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                    fps, (frame_width, frame_height))

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while vf.isOpened():
            ret, image = vf.read()
            if not ret:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image,
                                      results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.
                                      get_default_pose_landmarks_style())
            out_annotated.write(image)

        out_annotated.release()
        vf.release()

    st.video(annotated_filename)

    fig = make_plot(params_list)

    st.write(fig)

    st.text('')

    col1, col2 = st.columns(2)

    col1.header('Preparation ⏳')
    col1.write(exercices[response['prediction']]["Instructions_Preparation"])
    col1.text('')
    col1.header('Execution ✅')
    col1.write(exercices[response['prediction']]["Instructions_Execution"])
    col1.header('Tips 🗣')
    col1.write(exercices[response['prediction']]["Comments"])

    col2.header('To take it easy 🏝')
    col2.write(exercices[response['prediction']]["Easier"])
    col2.header('To push harder 🔥')
    col2.write(exercices[response['prediction']]["Harder"])

else:
    #Image for homepage
    image_intro = st.image('streamlit/intro.jpeg')

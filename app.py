import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import requests
import streamlit as st
import tempfile
from Coach_AI.data import json_to_df
from Coach_AI.exercices import exercices
from Coach_AI.utils import make_plot, download_model
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

emojis = {
    'cardio': '🫀',
    'pecs and arms': '💪🏻',
    'thighs': '🦵🏻',
    'glutes and calfs': '🍑',
    'back muscles': '🔙'
}


st.title('Coach AI')


#Instantiate Mediapipe Objects
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['Run on Video', 'Run on Webcam'])

if app_mode == 'Run on Video':
    st.sidebar.header('Upload your video :video_camera:')

    video = st.sidebar.file_uploader('Video File',
                                 type=["mp4", "mov", 'avi', 'asf', 'm4v'])

    if video:
        url = 'https://coachai-ujeungn6tq-ew.a.run.app/predict'
        data = {'video': video}

        response = requests.post(url, files=data).json()

        params_list = [np.array(element) for element in response['lists']]

        params_list.append(response['label'])

        st.header(
            f"Well done! You did **{response['reps']} {response['prediction']}s**! 🥇")

        target = exercices[response['prediction']]["Target_Muscle_Group"]

        st.subheader(f'You worked on your {target} {emojis[target]}!')

        st.subheader(f'You are one step closer to your goals! 🏆')

        st.sidebar.video(video)

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
        st.subheader('')
        st.subheader('**Your body markers** 📍')
        st.video(annotated_filename)
        st.subheader('')
        st.subheader('**Your body movement** 🤸🏻‍♂️')
        fig = make_plot(params_list)

        st.write(fig)

        st.subheader('')

        st.header('**Going further...🚀**')

        col1, col2 = st.columns(2)

        col1.subheader('Preparation ⏳')
        col1.write(exercices[response['prediction']]["Instructions_Preparation"])
        col1.text('')
        col1.subheader('Execution ✅')
        col1.write(exercices[response['prediction']]["Instructions_Execution"])
        col1.subheader('Tips 🗣')
        col1.write(exercices[response['prediction']]["Comments"])

        col2.subheader('To take it easy 🏝')
        col2.write(exercices[response['prediction']]["Easier"])
        col2.subheader('To push harder 🔥')
        col2.write(exercices[response['prediction']]["Harder"])

        st.subheader('Good form example 👀')

        col3, col4 = st.columns(2)

        col3.video(exercices[response['prediction']]["video_src"])

    else:
        #Image for homepage
        st.header('')
        image_intro = st.image('streamlit/intro.jpeg')

if app_mode == 'Run on Webcam':
    category = {
        0: 'Push Up',
        1: 'Jumping Jack',
        2: 'Squat',
        3: 'Lunge',
        4: 'Pull Up'
    }

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.landmarks = {}
            self.i = 0
            self.landmark_dict = {}
            self.data = []
            self.prediction = ''

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            frame_width = int(img.shape[1])
            frame_height = int(img.shape[0])
            j = 0
            with mp_pose.Pose(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as pose:
                results = pose.process(img)
                mp_drawing.draw_landmarks(img,
                                        results.pose_landmarks,
                                        mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing_styles.
                                        get_default_pose_landmarks_style())
                pose_landmarks = results.pose_landmarks
                if pose_landmarks != None:
                    for lmk in pose_landmarks.landmark:
                        self.landmarks[j] = [
                            lmk.x * frame_width, lmk.y * frame_height,
                            lmk.z * frame_width
                        ]
                        j += 1
                self.landmark_dict[self.i] = self.landmarks
                self.i += 1
                if len(self.landmark_dict) >= 20:
                    self.data.append(json_to_df(self.landmark_dict)[0])
                    print("--------self data [0]--------")
                    print(self.data[0])
                    cols = [
                        'min_d0', 'max_d0', 'min_d1', 'max_d1', 'min_d2', 'max_d2',
                        'min_d3', 'max_d3', 'min_d4', 'max_d4', 'min_d5', 'max_d5',
                        'min_d6', 'max_d6', 'min_d7', 'max_d7', 'min_d8', 'max_d8',
                        'min_d9', 'max_d9', 'min_d10', 'max_d10', 'min_d11',
                        'max_d11', 'min_d12', 'max_d12', 'min_a0', 'max_a0',
                        'min_a1', 'max_a1', 'min_a2', 'max_a2', 'min_a3', 'max_a3',
                        'min_a4', 'max_a4', 'min_a5', 'max_a5', 'mean_body_angle'
                    ]
                    X = pd.DataFrame(self.data[0])
                    X.columns = cols
                    print(X)
                    model = pickle.load(open('svc_coachai.pkl', "rb"))
                    results = model.predict(X)
                    print(results[0])
                    print(category[results[0]])
                    self.prediction = category[results[0]]
            cv2.rectangle(img, (0, 0), (225, 73), (255, 255, 255), -1)
            cv2.putText(img, f'You are doing a {self.prediction}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
            return img

    webrtc_streamer(key="Start Recording", video_processor_factory=VideoTransformer)


def load_css(filename):
    with open(filename) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

load_css('style.css')

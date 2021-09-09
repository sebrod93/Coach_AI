from requests.models import Response
import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import mediapipe as mp
import os
import requests
import tempfile
from PIL import Image

st.title('Coach AI')

st.sidebar.header('Upload your video')

video = st.sidebar.file_uploader('Video File',
                                 type=["mp4", "mov", 'avi', 'asf', 'm4v'])

if video:
    url = 'https://coachai-ujeungn6tq-ew.a.run.app/predict'
    data = {'video': video}
    x = requests.post(url, files=data)
    st.write(x.text)

    col1, col2 = st.columns(2)

    col1.header('lisa a la pression')

    col2.header('maxime a la pression')

    st.text('')

    col3, col4 = st.columns(2)

    col3.header('Seb a la pression')

    col4.header('maxime a la pression')

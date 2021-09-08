from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
from CoachAI.data import video_processing, json_to_df
from CoachAI.utils import download_model
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return dict(greeting="hello coach ai")

@app.post("/predict")
def predict(video: UploadFile = File(...)):
    
    current_directory = os.getcwd()
    video_directory = os.path.join(current_directory, r'videoUploads')
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)
   
    with open(f'videoUploads/{video.filename}', 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)
        json = video_processing(buffer.name)
        data = json_to_df(json)
        
        cols = ['min_d0', 'max_d0', 'min_d1', 'max_d1', 'min_d2', 'max_d2', 'min_d3', 'max_d3', 'min_d4', 'max_d4',
            'min_d5', 'max_d5', 'min_d6', 'max_d6', 'min_d7', 'max_d7', 'min_d8', 'max_d8', 'min_d9', 'max_d9',
            'min_d10', 'max_d10', 'min_d11', 'max_d11', 'min_d12', 'max_d12', 'min_a0', 'max_a0', 'min_a1', 'max_a1',
            'min_a2', 'max_a2', 'min_a3', 'max_a3', 'min_a4', 'max_a4', 'min_a5', 'max_a5','mean_body_angle']
        
        X = pd.DataFrame(data, columns =cols)

        model = download_model()
        
        results = model.predict(X)

        category = {0:'Push Up', 1:'Jumping Jack', 2:'Squat', 3:'Lunge', 4:'Pull Up'}
        
        print(results)
        return category[results[0]]
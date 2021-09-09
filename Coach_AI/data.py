import cv2
import mediapipe as mp
import numpy as np
import json
from Coach_AI.utils import dict_to_array, calculate_pairwise_distances, calculate_set_of_angles, calculate_body_angle

def video_processing(video):
    video_input = cv2.VideoCapture(video)

    #Initialize a dictionary to store landmark results for each frame
    landmark_dict = {}
    i = 0

    #Store the width and height for the current video
    frame_width = int(video_input.get(3))
    frame_height = int(video_input.get(4))

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        #Iterates through each frame in the current video
        while video_input.isOpened():

            ret, image = video_input.read()
            if not ret:
                # return 'pas de ret'
                break

            #Create an image of the current frmae
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #Stores the landmarks of the current frame
            pose_landmarks = results.pose_landmarks

            frame_dict = {}
            j = 0

            if pose_landmarks != None:
                for lmk in pose_landmarks.landmark:
                    frame_dict[j] = [lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                    j += 1

                landmark_dict[i] = frame_dict

            i += 1

    video_input.release()

    return landmark_dict

def json_to_df(json):
    data = []
    frames_distances = []
    frames_angles =[]
    body_angles=[]
    for index in range(0,len(json)):
        frame = json[index]
        coordinates = dict_to_array(frame)
        distances_list = calculate_pairwise_distances(coordinates)
        angles_list = calculate_set_of_angles(coordinates)
        body_angle = calculate_body_angle(coordinates, 'LAnkle', 'Nose')
        body_angles.append(body_angle)
        frames_distances.append(distances_list)
        frames_angles.append(angles_list)

    distances_array = np.array(frames_distances)
    joints_angles_array = np.array(frames_angles)
    min_angles = joints_angles_array.min(axis=0)
    max_angles = joints_angles_array.max(axis=0)

    angles_array = np.array(body_angles)
    min_distances = distances_array.min(axis=0)
    max_distances = distances_array.max(axis=0)

    mean_body_angle = angles_array.mean()
    row = []

    for position in range(0,len(min_distances)):
        row.append(min_distances[position])
        row.append(max_distances[position])

    for position in range (0,len(min_angles)) :
        row.append(min_angles[position])
        row.append(max_angles[position])

    row.append(mean_body_angle)
    data.append(row)

    return data, distances_array, joints_angles_array

# if __name__ == '__main__':

#     pass

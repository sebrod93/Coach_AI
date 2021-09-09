import numpy as np
import math
import os
import shutil
from google.cloud import storage
from Coach_AI.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION, LOCAL_MODEL_NAME
import pickle
import cv2
import mediapipe as mp
import tempfile


def download_model(rm=False):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{LOCAL_MODEL_NAME}"
    blob = bucket.blob(storage_location)
    blob.download_to_filename(LOCAL_MODEL_NAME)
    print("Model downloaded from GCP")

    model = pickle.load(open(LOCAL_MODEL_NAME,"rb"))

    if rm:
        os.remove(LOCAL_MODEL_NAME)

    return model

BODY_POINTS = {
        "Nose":  0,
        "LEyeIn": 1,
        "LEye": 2,
        "LEyeOut": 3,
        "REyeIn": 4,
        "REye": 5,
        "REyeOut": 6,
        "LEar": 7,
        "REar": 8,
        "LMouth": 9,
        "RMouth": 10,
        "LShoulder": 11,
        "RShoulder": 12,
        "LElbow": 13,
        "RElbow": 14,
        "LWrist": 15,
        "RWrist": 16,
        "LPinky": 17,
        "RPinky": 18,
        "LIndex": 19,
        "RIndex": 20,
        "LThumb": 21,
        "RThumb": 22,
        "LHip": 23,
        "RHip": 24,
        "LKnee": 25,
        "RKnee": 26,
        "LAnkle": 27,
        "RAnkle": 28,
        "LHeel": 29,
        "RHeel": 30,
        "LFoot": 31,
        "RFoot": 32
        }

def dict_to_array(dictionary):
    array = []
    for key, value in dictionary.items():
        array.append(value)

    return np.array(array)

def calculate_body_angle(coordinates, body_part1, body_part2):
    coordinates = coordinates[:, :2].copy()
    a = coordinates[BODY_POINTS[body_part1]]
    b = coordinates[BODY_POINTS[body_part2]]
    c = np.array([b[0],a[1]])
    vector = np.subtract(b,a)
    v1 = np.subtract(b,a)
    v2 = np.subtract(c,a)
    dot = v1.dot(v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(dot/(norm_v1*norm_v2))

    return math.degrees(angle_rad)

def get_pose_center(coordinates):
    '''Calculate the centre of the pose, define as the midpoint between point the neck and the mid hip'''
    mid_hip = (coordinates[BODY_POINTS['LHip']] + coordinates[BODY_POINTS['RHip']])/2
    neck = (coordinates[BODY_POINTS['LShoulder']] + coordinates[BODY_POINTS['RShoulder']])/2
    center = (mid_hip + neck)/2

    return center

def get_distance(coordinates, body_part1, body_part2, dimensions = '2D'):
    '''Calculates the distance between two body parts'''
    if dimensions == '2D':
        coordinates = coordinates[:, :2].copy()
    bp1 = coordinates[BODY_POINTS[body_part1]]
    bp2 = coordinates[BODY_POINTS[body_part2]]
    distance = ((bp1-bp2)*(bp1-bp2)).sum() ** 0.5

    return distance

def get_pose_size(coordinates, torso_size_multiplier=2):
    """Calculates pose size.
    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # This approach uses only 2D landmarks to compute pose size.
    coordinates = coordinates[:, :2].copy()

    # Hips center.
    left_hip = coordinates[BODY_POINTS['LHip']]
    right_hip = coordinates[BODY_POINTS['RHip']]
    hips = (left_hip + right_hip) * 0.5

    # Shoulders center.
    left_shoulder = coordinates[BODY_POINTS['LShoulder']]
    right_shoulder = coordinates[BODY_POINTS['RShoulder']]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)

    # Max dist to pose center.
    pose_center = get_pose_center(coordinates)
    max_dist = np.max(np.linalg.norm(coordinates - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)

def normalize_pose_landmarks(coordinates, torso_size_multiplier=2):
    """Normalizes landmarks translation and scale."""

    norm_coordinates = coordinates.copy()

    # Normalize translation.
    pose_center = get_pose_center(norm_coordinates)
    norm_coordinates -= pose_center

    # Normalize scale.
    pose_size = get_pose_size(norm_coordinates, torso_size_multiplier)
    norm_coordinates /= pose_size
    # Multiplication by 100 is not required, but makes it eaasier to debug.
    norm_coordinates *= 100

    return norm_coordinates

def get_angle(coordinates, body_part1, body_part2, body_part3, dimensions = '2D'):
    '''Calculates the angle between three body parts'''
    if dimensions == '2D':
        coordinates = coordinates[:, :2].copy()
    a = coordinates[BODY_POINTS[body_part1]]
    b = coordinates[BODY_POINTS[body_part2]]
    c = coordinates[BODY_POINTS[body_part3]]
    v1 = np.subtract(a,b)
    v2 = np.subtract(c,b)
    dot = v1.dot(v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(dot/(norm_v1*norm_v2))

    return math.degrees(angle_rad)

def calculate_pairwise_distances(coordinates, torso_size_multiplier=2):
    '''Calculates a set of distances for a coordinate system'''
    pairs = [('LShoulder', 'LWrist'), ('RShoulder', 'RWrist'), ('RHip', 'RAnkle'), ('LHip', 'LAnkle'), ('RWrist', 'LWrist'),
             ('LAnkle', 'RAnkle'), ('RHip', 'RWrist'), ('LHip', 'LWrist'), ('LWrist', 'LAnkle'), ('RWrist', 'RAnkle'),
             ('RKnee', 'LKnee'), ('RHip', 'LKnee'), ('LHip', 'RKnee')]

    distances_list = []

    for pair in pairs:
        norm_coordinates = normalize_pose_landmarks(coordinates)
        distance = round(get_distance(norm_coordinates, pair[0], pair[1]),2)
        distances_list.append(distance)

    return distances_list

def calculate_set_of_angles(coordinates):
    '''Calculates a set of angles for a coordinates system'''
    joints = [('LShoulder', 'LElbow', 'LWrist'), ('RShoulder', 'RElbow', 'RWrist'), ('LShoulder', 'LHip', 'LKnee'),
              ('RShoulder', 'RHip', 'RKnee'), ('LHip', 'LKnee', 'LAnkle'), ('RHip', 'RKnee', 'RAnkle')]
    angles_list = []
    for joint in joints:
        angle = round(get_angle(coordinates, joint[0], joint[1], joint[2]),2)
        angles_list.append(angle)

    return angles_list

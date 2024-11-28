import os
import pandas as pd
import mediapipe as mp
import cv2
import pickle
import numpy as np


from data_processing import direct_csv_to_video, video_files
from engine import Brady_Signs

# Mediapipe Holistic setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Directory to save pickle files
output_dir = "brady_sign_pickles"
os.makedirs(output_dir, exist_ok=True)

def extract_holistic_keypoints(video_path):
    """Extract holistic keypoints (pose, face, hands) from a video."""
    cap = cv2.VideoCapture(video_path)
    keypoints_seq = []

    # Define the fixed shape for each frame (33 pose, 21 left hand, 21 right hand, 468 face)
    num_pose_landmarks = 33
    num_hand_landmarks = 21
    num_face_landmarks = 468

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb_frame)

        # Initialize an empty numpy array for keypoints with a fixed size
        keypoints = np.zeros((33 * 4 + 21 * 3 + 21 * 3 + 468 * 3))  # Total features for pose, hands, face

        # Pose landmarks (33 landmarks, each with 4 values: x, y, z, visibility)
        if result.pose_landmarks:
            pose_data = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]).flatten()
            keypoints[:33 * 4] = pose_data  # Assign pose data to the first part of the array

        # Left hand landmarks (21 landmarks, each with 3 values: x, y, z)
        if result.left_hand_landmarks:
            left_hand_data = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark]).flatten()
            keypoints[33 * 4:33 * 4 + 21 * 3] = left_hand_data  # Assign left hand data

        # Right hand landmarks (21 landmarks, each with 3 values: x, y, z)
        if result.right_hand_landmarks:
            right_hand_data = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark]).flatten()
            keypoints[33 * 4 + 21 * 3:33 * 4 + 21 * 3 * 2] = right_hand_data  # Assign right hand data

        # Face landmarks (468 landmarks, each with 3 values: x, y, z)
        if result.face_landmarks:
            face_data = np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks.landmark]).flatten()
            keypoints[33 * 4 + 21 * 3 * 2:] = face_data  # Assign face data

        # Append the fixed-size keypoints array to the sequence
        keypoints_seq.append(keypoints)

    cap.release()
    return keypoints_seq


def process_brady_signs(Brady_Signs):
    """Process all videos from the Brady_Signs DataFrame."""
    for index in range(2508,len(Brady_Signs)):
        try:
            # Get the snippet path directly from your implemented function
            snippet_path = direct_csv_to_video(
                Brady_Signs['full video file'][index],
                str(Brady_Signs['start frame of the sign (relative to full videos)'][index]),
                str(Brady_Signs['end frame of the sign (relative to full videos)'][index])
            )
            label = Brady_Signs['Class Label'][index]
            id = Brady_Signs['Video ID number'][index]

            # Process video to extract keypoints
            print(f"Processing video: {snippet_path} with label: {label}")
            keypoints_seq = extract_holistic_keypoints(snippet_path)

            # Sanitize the snippet_path for filename use
            sanitized_snippet_path = snippet_path.replace('/', '_').replace('\\', '_')

            # Create pickle filename
            filename = f"{id}_{label}_{sanitized_snippet_path}.pkl"
            output_path = os.path.join(output_dir, filename)

            # Save keypoints to pickle
            with open(output_path, 'wb') as f:
                pickle.dump(keypoints_seq, f)
            print(f"Saved features to: {output_path}")
            print(f"Processed video index: {index}")

        except FileNotFoundError:
            print(f"Video file not found for index {index}. Skipping...")
            continue
        except Exception as e:
            print(f"An error occurred for index {index}: {e}. Skipping...")
            continue

# Example usage
process_brady_signs(Brady_Signs)

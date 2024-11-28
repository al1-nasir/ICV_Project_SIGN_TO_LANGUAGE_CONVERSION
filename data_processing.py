import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Specify the directory in WSL format (Linux path)
main_dir = "/mnt/c/Users/Ali Nasir/ICV_batches"

# List to store all the video file paths
video_files = []


def direct_csv_to_video(full_video_name, starting_frame, ending_frame, main_dir=main_dir):
    # Split the filename on underscores
    parts = full_video_name.split('_')

    # Extract the date part (first three parts) e.g. ASL_2011_06_08
    date_part = '_'.join(parts[1:4])  # ASL_2011_06_08_Brady

    # Extract the person's name (fourth part) e.g. Brady
    person_name = parts[4]

    # Extract the camera information (everything after the last dash)
    camera_info = full_video_name.split('-')[-1]  # camera1.mov

    scene_number = re.search(r'scene(\d+)', parts[5]).group(1)

    # Construct the desired output string in correct order
    video_name = f"{person_name}-session-{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{person_name}-scene-{scene_number}-{starting_frame}-{ending_frame}-{camera_info}"

    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if video_name in file:
                print(f"Found file: {file}")
                return os.path.join(root, file)



def filter_and_save_brady_rows(input_csv, output_csv, column_name):
    """
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
        column_name (str): The name of the column to search for 'Brady'.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Create a boolean mask where the column contains 'Brady'
    mask = df[column_name].str.contains('Brady', case=False, na=False)

    # Use iloc to select rows where the mask is True
    filtered_df = df.iloc[mask.values]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)

    print(f"Filtered data saved to {output_csv}")


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    # Read and display frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            print("end of the video or failed to read frame.")
            break

        cv2.imshow('Video', frame)
        # Exit the video window when 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

def get_all_files_from_directory(directory):
    video_files = []
    for root, dirs, files in os.walk(main_dir):
        print(f"Currently looking in: {root}")
        for file in files:
            # print(f"Found file: {file}")
            if file.endswith(('.mp4', '.mov', '.avi', '.mkv')):
                # Get the full path of the video file
                full_path = os.path.join(root, file)
                video_files.append(full_path)
    return video_files

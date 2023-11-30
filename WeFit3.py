import streamlit as st
import cv2
import mediapipe as mp
import math
import datetime
import csv
import time

# Initialize the MediaPipe library
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Global variables
squats_count = 0
in_squat = False
reference_angle = 40  # Initial reference angle
level = "Easy"
squat_angle = 0  # Add the squat angle variable
training_data = []
start_time = time.time()
interval = 300  # 5 minutes in seconds

# Function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    a = calculate_distance(point1, point2)
    b = calculate_distance(point2, point3)
    c = calculate_distance(point1, point3)
    angle_rad = math.acos((a**2 + b**2 - c**2) / (2 * a * b))
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# Function to check squat posture
def check_squats(landmarks):
    global squats_count, in_squat, squat_angle

    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

    angle = calculate_angle(left_knee, hip, left_ankle)

    # Adjust the angle variable accordingly
    squat_angle = 180 - int(angle)  # Invert the angle to go from top to bottom

    if squat_angle < reference_angle:
        if not in_squat:
            in_squat = True
            squats_count += 1
    else:
        in_squat = False

    return squats_count, squat_angle

# Streamlit App
def main():
    global squats_count, reference_angle, level, squat_angle, training_data, start_time

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Detect body pose joints
        results_pose = pose.process(frame)

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark

            # Check squat posture
            squats_count, squat_angle = check_squats(landmarks)

        # Display the resulting frame
        st.image(frame, channels="BGR", use_column_width=True)

        # Add date and time to the frame
        current_time = datetime.datetime.now()
        formatted_date = current_time.strftime("%d/%m/%Y")
        formatted_time = current_time.strftime("%H:%M:%S")
        
        # Display the information in the Streamlit app
        st.text(f"Reps: {squats_count:.0f}")
        st.text(f"Date: {formatted_date}")
        st.text(f"Time: {formatted_time}")
        st.text(f"Level: {level}")
        st.text(f"Squat Angle: {squat_angle:.2f}")

        # Check if it's time to create a new CSV entry
        current_time = time.time()
        if current_time - start_time >= interval:
            training_data.append([squats_count, formatted_date, formatted_time, level, squat_angle])
            start_time = current_time

        # Exit the loop if the user interrupts the script manually
        if st.button("Interrupt Script"):
            break

    # Release the webcam and close the window
    cap.release()

if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import math
import tkinter as tk
from PIL import Image, ImageTk

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def detect_pose(landmarks):
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    avg_hip_y = (left_hip.y + right_hip.y) / 2
    avg_knee_y = (left_knee.y + right_knee.y) / 2

    # Detecting "T-Pose"
    if abs(left_wrist.y - left_elbow.y) < 0.1 and abs(right_wrist.y - right_elbow.y) < 0.1 and abs(left_wrist.y - left_shoulder.y) < 0.1 and abs(right_wrist.y - right_shoulder.y) < 0.1:
        return "T-Pose"
    
    # Detecting "Heart"
    if abs(left_wrist.x - right_wrist.x) < 0.3 and left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y and left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y and not abs(left_wrist.x - left_elbow.x) < 0.1 and not abs(right_wrist.x - right_elbow.x) < 0.1:
        return "Heart"
    
    # Detecting "Hands Up"
    if left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y and left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
        return "Hands Up"

    # Detecting "Waving"
    if (left_wrist.y < left_elbow.y or right_wrist.y < right_elbow.y) and (left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y):
        return "Waving"

    # Detecting "Folded Hands"
    if abs(left_wrist.x - right_wrist.x) < 0.05 and left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y and left_wrist.y > left_shoulder.y:
        return "Folded Hands"

    # Detecting "Arms Crossed"
    if left_wrist.x > right_shoulder.x and right_wrist.x < left_shoulder.x and left_elbow.y > left_wrist.y and right_elbow.y > right_wrist.y:
        return "Arms Crossed"

    # Detecting "Bending Down"
    if abs(avg_shoulder_y - avg_hip_y) <= 0.2 and avg_hip_y < avg_knee_y:
        return "Bending Down"
    
    # Detecting "Sitting"
    if avg_hip_y > avg_knee_y and abs(left_hip.y - right_hip.y) < 0.5:
        return "Sitting"
    
    # Detecting "Standing"
    if abs(left_wrist.y - left_hip.y) < 0.1 and abs(right_wrist.y - right_hip.y) < 0.1 and left_elbow.x < left_wrist.x and right_elbow.x > right_wrist.x:
        return "Standing"

    # Detecting "Hands on hips"
    if abs(avg_shoulder_y - avg_hip_y) > 0.2 and avg_hip_y < avg_knee_y:
        return "Hands on Hips"

    return "Unknown Pose"

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_type = detect_pose(results.pose_landmarks)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        border_color = (0, 0, 0)      # Black
        text_position = (10, 30)

        for i in range(3):
            cv2.putText(frame, pose_type, (text_position[0] - i, text_position[1] - i), font, font_scale, border_color, font_thickness + i, cv2.LINE_AA)

        cv2.putText(frame, pose_type, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    window_width = max(1, root.winfo_width())
    window_height = max(1, root.winfo_height())
    frame_height, frame_width, _ = frame.shape
    aspect_ratio = frame_width / frame_height

    if window_width / window_height > aspect_ratio:
        new_height = window_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = window_width
        new_height = int(new_width / aspect_ratio)

    if new_width > 0 and new_height > 0:
        resized_frame = cv2.resize(frame, (new_width, new_height))
    else:
        resized_frame = frame

    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(image=frame_pil)
    label.imgtk = frame_tk
    label.configure(image=frame_tk)

    label.after(10, update_frame)

root = tk.Tk()
root.title("Pose Recognition")
root.geometry("800x600")

label = tk.Label(root)
label.pack(fill=tk.BOTH, expand=True)

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
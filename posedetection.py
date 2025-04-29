import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk

# Setup MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Pose detection function
def detect_pose(landmarks):
    # Get relevant landmarks
    wrist_positions = {
        'left_wrist': landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST],
        'right_wrist': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST],
        'left_elbow': landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
        'right_elbow': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
        'left_shoulder': landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
        'right_shoulder': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        'left_hip': landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
        'right_hip': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
        'left_knee': landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
        'right_knee': landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    }

    # Calculate average positions
    avg_shoulder_y = (wrist_positions['left_shoulder'].y + wrist_positions['right_shoulder'].y) / 2
    avg_hip_y = (wrist_positions['left_hip'].y + wrist_positions['right_hip'].y) / 2
    avg_knee_y = (wrist_positions['left_knee'].y + wrist_positions['right_knee'].y) / 2

    # Helper functions for pose detection
    def is_wrist_near_shoulder(wrist, shoulder, threshold=0.1):
        return abs(wrist.y - shoulder.y) < threshold

    def is_wrist_near_elbow(wrist, elbow, threshold=0.1):
        return abs(wrist.y - elbow.y) < threshold

    # Pose detection logic
    if is_wrist_near_elbow(wrist_positions['left_wrist'], wrist_positions['left_elbow']) and \
       is_wrist_near_elbow(wrist_positions['right_wrist'], wrist_positions['right_elbow']) and \
       is_wrist_near_shoulder(wrist_positions['left_wrist'], wrist_positions['left_shoulder']) and \
       is_wrist_near_shoulder(wrist_positions['right_wrist'], wrist_positions['right_shoulder']):
        return "T-Pose"
    
    if abs(wrist_positions['left_wrist'].x - wrist_positions['right_wrist'].x) < 0.3 and \
       wrist_positions['left_wrist'].y < wrist_positions['left_elbow'].y and \
       wrist_positions['right_wrist'].y < wrist_positions['right_elbow'].y and \
       wrist_positions['left_wrist'].y < wrist_positions['left_shoulder'].y and \
       wrist_positions['right_wrist'].y < wrist_positions['right_shoulder'].y:
        return "Heart"
    
    if wrist_positions['left_wrist'].y < wrist_positions['left_elbow'].y and \
       wrist_positions['right_wrist'].y < wrist_positions['right_elbow'].y and \
       wrist_positions['left_wrist'].y < wrist_positions['left_shoulder'].y and \
       wrist_positions['right_wrist'].y < wrist_positions['right_shoulder'].y:
        return "Hands Up"
    
    if (wrist_positions['left_wrist'].y < wrist_positions['left_elbow'].y or \
        wrist_positions['right_wrist'].y < wrist_positions['right_elbow'].y) and \
       (wrist_positions['left_wrist'].y < wrist_positions['left_shoulder'].y or \
        wrist_positions['right_wrist'].y < wrist_positions['right_shoulder'].y):
        return "Waving"
    
    if abs(wrist_positions['left_wrist'].x - wrist_positions['right_wrist'].x) < 0.05 and \
       wrist_positions['left_wrist'].y < wrist_positions['left_elbow'].y and \
       wrist_positions['right_wrist'].y < wrist_positions['right_elbow'].y and \
       wrist_positions['left_wrist'].y > wrist_positions['left_shoulder'].y:
        return "Folded Hands"
    
    if wrist_positions['left_wrist'].x > wrist_positions['right_shoulder'].x and \
       wrist_positions['right_wrist'].x < wrist_positions['left_shoulder'].x and \
       wrist_positions['left_elbow'].y > wrist_positions['left_wrist'].y and \
       wrist_positions['right_elbow'].y > wrist_positions['right_wrist'].y:
        return "Arms Crossed"
    
    if abs(avg_shoulder_y - avg_hip_y) <= 0.2 and avg_hip_y < avg_knee_y:
        return "Bending Down"
    
    if avg_hip_y > avg_knee_y and abs(wrist_positions['left_hip'].y - wrist_positions['right_hip'].y) < 0.5:
        return "Sitting"
    
    if abs(wrist_positions['left_wrist'].y - wrist_positions['left_hip'].y) < 0.1 and \
       abs(wrist_positions['right_wrist'].y - wrist_positions['right_hip'].y) < 0.1 and \
       wrist_positions['left_elbow'].x < wrist_positions['left_wrist'].x and \
       wrist_positions['right_elbow'].x > wrist_positions['right_wrist'].x:
        return "Standing"

    return "Unknown Pose"

# Tkinter GUI update
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

        # Display pose type on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        border_color = (0, 0, 0)
        text_position = (10, 30)
        cv2.putText(frame, pose_type, text_position, font, 1, text_color, 2, cv2.LINE_AA)

    # Resize frame to fit window aspect ratio
    window_width, window_height = root.winfo_width(), root.winfo_height()
    frame_height, frame_width, _ = frame.shape
    aspect_ratio = frame_width / frame_height
    if window_width / window_height > aspect_ratio:
        new_height = window_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = window_width
        new_height = int(new_width / aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(image=frame_pil)
    
    label.imgtk = frame_tk
    label.configure(image=frame_tk)

    label.after(10, update_frame)

# Tkinter setup
root = tk.Tk()
root.title("Pose Recognition")
root.geometry("800x600")

label = tk.Label(root)
label.pack(fill=tk.BOTH, expand=True)

# Start webcam frame updates
update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()

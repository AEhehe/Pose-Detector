# Pose-Detector
Python app for real-time pose detection using OpenCV and MediaPipe. It identifies various human poses like "T-Pose" and "Hands Up" from a webcam feed and displays them with a Tkinter GUI. Features dynamic video resizing and pose labeling.
## Overview
**Pose-Detector** is a Python application for real-time pose recognition using OpenCV and MediaPipe. It captures video from a webcam and identifies various human poses such as **"T-Pose," "Hands Up," "Waving,"** and more. The application displays the detected pose labels on the video feed with a responsive Tkinter GUI.
## Features
- **Real-Time Pose Detection**: Uses MediaPipe for accurate pose landmark detection.
- **Pose Classification**: Recognizes multiple predefined poses based on joint positions and angles.
- **Dynamic Video Display**: Adjusts video feed size within a Tkinter window while maintaining aspect ratio.
- **Custom Pose Labels**: Displays detected pose names on the video feed for clear identification.
## Installation
To get started with Pose-Detector, follow these steps:
1. **Clone the repository:**
    ```
    git clone https://github.com/iseerahul/Pose-Detector.git
    ```
2. **Navigate to the project directory:**
    ```
    cd Pose-Detector
    ```
3. **Install the required dependencies:**
    ```
    pip install opencv-python mediapipe pillow
    ```
## Usage
Run the application with Python:
```
python pose_detector.py
```
A window will open showing the webcam feed with detected poses labeled in real-time.
## Contributing
**Contributions are welcome!** If you have ideas for new features or improvements, please open an issue or submit a pull request.
## License
This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
## Acknowledgments
- **OpenCV**: Library for computer vision tasks.
- **MediaPipe**: Framework for pose detection.
- **Tkinter**: GUI toolkit for Python.

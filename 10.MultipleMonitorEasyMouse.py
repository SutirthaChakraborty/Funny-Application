# pip install mediapipe cv2 pyautogui screeninfo numpy pyttsx3 pynput pystray PIL
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from screeninfo import get_monitors
import math
import pyttsx3
from pynput import keyboard
import threading
import time
import sys


class GazeControlledMouse:
    def __init__(self):
        # Initialize MediaPipe Face Mesh with Iris
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # Set the frame size for performance optimization
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)

        # Get monitor information
        self.monitors = get_monitors()

        # Calibration data structure
        self.calibration_data = []

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speech rate

        # For tracking the last monitor to prevent redundant movements
        self.last_monitor = None
        self.last_positions = {idx: None for idx in range(len(self.monitors))}

        # Lock for thread-safe operations
        self.lock = threading.Lock()

    def speak(self, text):
        """Function to speak the given text."""
        self.engine.say(text)
        self.engine.runAndWait()

    def get_eye_landmarks(self, face_landmarks, eye='left'):
        """
        Extracts eye landmarks for the specified eye.

        Args:
            face_landmarks: MediaPipe face landmarks.
            eye (str): 'left' or 'right'.

        Returns:
            List of tuples representing (x, y) coordinates of eye landmarks.
        """
        if eye == 'left':
            # Left eye indices based on MediaPipe Face Mesh
            indices = [
                33,
                160,
                158,
                133,
                153,
                144,
                163,
                7,
                246,
                161,
                160,
                159,
                158,
                157,
                173,
                133,
                155,
                154,
                153,
                145,
                144,
                163,
                7,
                33,
            ]
        else:
            # Right eye indices
            indices = [
                362,
                385,
                387,
                263,
                373,
                380,
                381,
                382,
                384,
                398,
                382,
                381,
                380,
                373,
                390,
                263,
                373,
                374,
                380,
                381,
                382,
                384,
                385,
                362,
            ]
        eye_landmarks = []
        for idx in indices:
            if idx < len(face_landmarks.landmark):
                x = face_landmarks.landmark[idx].x
                y = face_landmarks.landmark[idx].y
                eye_landmarks.append((x, y))
            else:
                # Handle missing landmarks gracefully
                eye_landmarks.append((0, 0))
        return eye_landmarks

    def get_iris_landmarks(self, face_landmarks, eye='left'):
        """
        Extracts iris landmarks for the specified eye.

        Args:
            face_landmarks: MediaPipe face landmarks.
            eye (str): 'left' or 'right'.

        Returns:
            List of tuples representing (x, y) coordinates of iris landmarks.
        """
        if eye == 'left':
            # Left iris indices based on MediaPipe Face Mesh
            iris_indices = [468, 469, 470, 471, 472, 473]
        else:
            # Right iris indices
            iris_indices = [473, 474, 475, 476, 477, 478]
        iris_landmarks = []
        for idx in iris_indices:
            if idx < len(face_landmarks.landmark):
                x = face_landmarks.landmark[idx].x
                y = face_landmarks.landmark[idx].y
                iris_landmarks.append((x, y))
            else:
                # Handle missing landmarks gracefully
                iris_landmarks.append((0, 0))
        return iris_landmarks

    def get_centroid(self, landmarks, image_width, image_height):
        """
        Computes the centroid of given landmarks.

        Args:
            landmarks (list): List of (x, y) tuples.
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            Tuple (x, y) representing the centroid in pixel coordinates, or None if no valid landmarks.
        """
        xs = [point[0] for point in landmarks if point != (0, 0)]
        ys = [point[1] for point in landmarks if point != (0, 0)]
        if not xs or not ys:
            return None
        centroid = (np.mean(xs), np.mean(ys))
        # Convert to pixel coordinates
        centroid_px = (int(centroid[0] * image_width), int(centroid[1] * image_height))
        return centroid_px

    def align_face(
        self,
        image,
        landmarks,
        desired_left_eye=(0.35, 0.35),
        desired_face_width=256,
        desired_face_height=None,
    ):
        """
        Aligns a face within an image using eye landmarks.

        Args:
            image (np.ndarray): The input image as a NumPy array (BGR format).
            landmarks (list): List of landmark points as tuples (x, y).
            desired_left_eye (tuple): Desired position of the left eye in the aligned face.
            desired_face_width (int): Desired width of the aligned face.
            desired_face_height (int): Desired height of the aligned face (default is None, will be equal to desired_face_width).

        Returns:
            tuple: A tuple containing the aligned face image (np.ndarray) and the transformation matrix (M).
        """
        if desired_face_height is None:
            desired_face_height = desired_face_width

        # The indices for the left and right eye centers.
        left_eye_idx = 130  # Adjusted to match the landmarks used
        right_eye_idx = 359

        if left_eye_idx >= len(landmarks) or right_eye_idx >= len(landmarks):
            return image, None  # Cannot align without proper eye landmarks

        # Extract the left and right eye (x, y) coordinates.
        left_eye_center = landmarks[left_eye_idx]
        right_eye_center = landmarks[right_eye_idx]

        # Compute the angle between the eye centroids.
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Calculate the desired right eye x-coordinate based on the desired x-coordinate of the left eye.
        desired_right_eye_x = 1.0 - desired_left_eye[0]

        # Determine the scale of the new resulting image by taking the ratio of the distance
        # between eyes in the current image to the ratio of distance in the desired image.
        dist = np.sqrt((dX**2) + (dY**2))
        desired_dist = (desired_right_eye_x - desired_left_eye[0]) * desired_face_width
        scale = desired_dist / dist if dist != 0 else 1.0

        # Compute center (x, y)-coordinates between the two eyes in the input image.
        eyes_center = (
            (left_eye_center[0] + right_eye_center[0]) // 2,
            (left_eye_center[1] + right_eye_center[1]) // 2,
        )

        # Grab the rotation matrix for rotating and scaling the face.
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Update the translation component of the matrix.
        tX = desired_face_width * 0.5
        tY = desired_face_height * desired_left_eye[1]
        M[0, 2] += tX - eyes_center[0]
        M[1, 2] += tY - eyes_center[1]

        # Apply the affine transformation.
        (w, h) = (desired_face_width, desired_face_height)
        aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return aligned_face, M

    def transform_landmarks(self, landmarks, M):
        """
        Transforms a list of landmarks using a given transformation matrix.

        Args:
            landmarks (list): List of landmark points as tuples (x, y).
            M (np.ndarray): Transformation matrix.

        Returns:
            list: Transformed landmark points as tuples (x, y).
        """
        transformed_landmarks = []
        for landmark in landmarks:
            x, y = landmark
            transformed_point = np.dot(M, np.array([x, y, 1]))
            transformed_landmarks.append(
                (int(transformed_point[0]), int(transformed_point[1]))
            )
        return transformed_landmarks

    def calibrate(self):
        """
        Performs the calibration phase by prompting the user to look at each monitor's center and recording gaze points.
        """
        self.speak("Starting calibration.")
        # print("Starting calibration...")
        calibration_points = len(self.monitors)
        for idx, monitor in enumerate(self.monitors):
            # Inform the user which monitor to look at
            message = f"Please look at the center of monitor {idx + 1}."
            self.speak(message)
            # print(message)
            time.sleep(2)  # Allow time for the user to focus

            # Collect multiple gaze points for better accuracy
            gaze_points = []
            num_samples = 5
            for _ in range(num_samples):
                ret, frame = self.cap.read()
                if not ret:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    # Convert landmarks to a list of tuples (x, y)
                    points = [
                        (int(p.x * frame.shape[1]), int(p.y * frame.shape[0]))
                        for p in face_landmarks.landmark
                    ]

                    # Align the face using the landmarks
                    aligned_face, M = self.align_face(frame, points)
                    if M is None:
                        continue

                    # Transform landmarks
                    transformed_landmarks = self.transform_landmarks(points, M)

                    # Extract iris landmarks from transformed landmarks
                    try:
                        left_iris = transformed_landmarks[468]
                        right_iris = transformed_landmarks[473]
                    except IndexError:
                        continue

                    # Compute the average gaze point based on iris positions
                    avg_gaze = (
                        (left_iris[0] + right_iris[0]) / 2,
                        (left_iris[1] + right_iris[1]) / 2,
                    )
                    gaze_points.append(avg_gaze)
                time.sleep(0.2)  # Short delay between samples

            if gaze_points:
                # Average the collected gaze points
                avg_gaze = (
                    np.mean([point[0] for point in gaze_points]),
                    np.mean([point[1] for point in gaze_points]),
                )
                self.calibration_data.append((avg_gaze, idx))
                self.speak(f"Calibrated monitor {idx + 1}.")
                # print(f"Calibrated monitor {idx + 1}.")
            else:
                self.speak(
                    f"Iris landmarks not detected properly for monitor {idx + 1}. Skipping calibration."
                )
                # print(
                #     f"Iris landmarks not detected properly for monitor {idx + 1}. Skipping calibration."
                # )
            time.sleep(1)
        self.speak("Calibration completed.")
        # print("Calibration completed.")

    def map_gaze_to_monitor(self, gaze, frame_width, frame_height):
        """
        Maps the average gaze point to the closest calibrated monitor.

        Args:
            gaze (tuple): Average gaze point (x, y).
            frame_width (int): Width of the frame.
            frame_height (int): Height of the frame.

        Returns:
            int or None: Index of the selected monitor or None if no calibration data exists.
        """
        if gaze is None:
            return None
        # Find the closest calibration data point
        min_distance = float('inf')
        selected_monitor = None
        for data in self.calibration_data:
            calib_gaze, monitor_idx = data
            distance = math.hypot(gaze[0] - calib_gaze[0], gaze[1] - calib_gaze[1])
            if distance < min_distance:
                min_distance = distance
                selected_monitor = monitor_idx
        return selected_monitor

    def move_mouse_to_position(self, x, y):
        """
        Moves the mouse cursor to the specified (x, y) position instantly.

        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.
        """
        pyautogui.moveTo(x, y, duration=0)  # duration=0 for instant movement

    def on_activate_exit(self):
        """
        Callback function when the exit hotkey is activated.
        """
        self.speak("Exiting application.")
        # print("Ctrl+Escape detected. Exiting application.")
        with self.lock:
            self.cap.release()
            sys.exit()

    def listen_for_exit_hotkey(self):
        """
        Sets up a global hotkey listener for Ctrl + Escape using pynput.
        """
        with keyboard.GlobalHotKeys(
            {'<ctrl>+<esc>': self.on_activate_exit}
        ) as listener:
            listener.join()

    def start_hotkey_listener(self):
        """
        Starts the hotkey listener in a separate daemon thread.
        """
        hotkey_thread = threading.Thread(
            target=self.listen_for_exit_hotkey, daemon=True
        )
        hotkey_thread.start()

    def run(self):
        """Main method to run the gaze-controlled mouse application."""
        self.start_hotkey_listener()
        self.calibrate()
        self.speak("Starting gaze tracking. Press Control Escape to quit.")
        # print("Starting gaze tracking. Press Ctrl+Escape to quit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.speak("Unable to capture frame from camera.")
                # print("Unable to capture frame from camera.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                # Convert landmarks to a list of tuples (x, y)
                points = [
                    (int(p.x * frame.shape[1]), int(p.y * frame.shape[0]))
                    for p in face_landmarks.landmark
                ]

                # Align the face using the landmarks
                aligned_face, M = self.align_face(frame, points)
                if M is None:
                    # print("Face alignment failed. Skipping frame.")
                    continue

                # Transform landmarks
                transformed_landmarks = self.transform_landmarks(points, M)

                # Extract iris landmarks from transformed landmarks
                try:
                    left_iris = transformed_landmarks[468]
                    right_iris = transformed_landmarks[473]
                except IndexError:
                    # print("Iris landmarks not detected properly. Skipping frame.")
                    continue

                # Compute the average gaze point based on iris positions
                avg_gaze = (
                    (left_iris[0] + right_iris[0]) / 2,
                    (left_iris[1] + right_iris[1]) / 2,
                )

                monitor_idx = self.map_gaze_to_monitor(
                    avg_gaze, frame.shape[1], frame.shape[0]
                )

                if monitor_idx is not None:
                    if monitor_idx != self.last_monitor:
                        # If switching monitors, save the current position of the previous monitor
                        if self.last_monitor is not None:
                            current_pos = pyautogui.position()
                            self.last_positions[self.last_monitor] = current_pos
                            # print(
                            #     f"Saved position {current_pos} for monitor {self.last_monitor + 1}"
                            # )

                        # Determine where to move the cursor
                        if self.last_positions[monitor_idx] is not None:
                            target_x, target_y = self.last_positions[monitor_idx]
                            # print(
                            #     f"Moving to saved position ({target_x}, {target_y}) on monitor {monitor_idx + 1}"
                            # )
                        else:
                            # If no saved position, move to the center
                            monitor = self.monitors[monitor_idx]
                            target_x = monitor.x + monitor.width // 2
                            target_y = monitor.y + monitor.height // 2
                            # print(
                            #     f"No saved position. Moving to center ({target_x}, {target_y}) on monitor {monitor_idx + 1}"
                            # )

                        # Move the mouse to the target position instantly
                        self.move_mouse_to_position(target_x, target_y)

                        # Update last_monitor
                        self.last_monitor = monitor_idx

            else:
                # print("Face not detected.")
                pass

    def cleanup(self):
        """Releases camera resources."""
        self.cap.release()


if __name__ == "__main__":
    app = GazeControlledMouse()
    try:
        app.run()
    except KeyboardInterrupt:
        app.cleanup()
        sys.exit()

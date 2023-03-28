# This Python code captures real-time webcam video and uses the MediaPipe Pose model to detect the user's head rotation. 
# Based on the head rotation, it adjusts the volume of two different audio channels (left and right) to create a crossfading effect, 
# making it seem like the sound source is in front of the listener.

import cv2
import mediapipe as mp
import numpy as np
import time
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


balance = 0
current_frame = 0

def callback(outdata, frames, time, status):
    global current_frame

    left_data = left_wave[current_frame:current_frame + frames, :]
    right_data = right_wave[current_frame:current_frame + frames, :]

    left_volume = max(min(0.5 - balance / 2, 1), 0)
    right_volume = max(min(0.5 + balance / 2, 1), 0)

    outdata[:, 0] = left_data[:, 0] * left_volume
    outdata[:, 1] = right_data[:, 0] * right_volume

    current_frame += frames
    if current_frame >= len(left_wave):
        current_frame = 0


# Load wav files
left_wav_file = '1.wav'
right_wav_file = '2.wav'

left_wave, left_sr = sf.read(left_wav_file, dtype='float32', always_2d=True)
right_wave, right_sr = sf.read(right_wav_file, dtype='float32', always_2d=True)

# Initialize audio stream
blocksize = 1024

stream = sd.OutputStream(samplerate=left_sr, channels=2, dtype='float32', blocksize=blocksize, callback=callback)
stream.start()


def calculate_head_rotation(left_ear, right_ear, nose_tip):
    left_ear_to_nose = np.linalg.norm(nose_tip - left_ear)
    right_ear_to_nose = np.linalg.norm(nose_tip - right_ear)

    if left_ear_to_nose == 0 or right_ear_to_nose == 0:
        return 0
    else:
        ratio = (right_ear_to_nose / left_ear_to_nose) - 1
        return ratio


# Initialize webcam
cap = cv2.VideoCapture(0)

# Set the desired FPS
desired_fps = 15
frame_time = 1 / desired_fps
ema_alpha = 0.1  # You can adjust this value for more or less smoothing (0 < ema_alpha < 1)

try:
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(frame_rgb)

        if pose_result.pose_landmarks:
            left_ear = np.array([pose_result.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,
                                    pose_result.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y,
                                    pose_result.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].z])

            right_ear = np.array([pose_result.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x,
                                    pose_result.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y,
                                    pose_result.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].z])

            nose_tip = np.array([pose_result.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                                    pose_result.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                                    pose_result.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z])

            rotation = calculate_head_rotation(left_ear, right_ear, nose_tip) 
            balance = (1 - ema_alpha) * balance + ema_alpha * rotation*10
            
            # Convert the rotation value to degrees and format as a string
            angle_text = f"Angle: {rotation * 90:.2f} degree, Balance : {balance:.2f}"

            # Display the angle on the frame
            cv2.putText(frame, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            left_volume = max(min(0.5 - balance / 2, 1), 0)
            right_volume = max(min(0.5 + balance / 2, 1), 0)


        # Display the webcam feed
        cv2.imshow('Webcam', frame)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Limit the FPS
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_time:
            time.sleep(frame_time - elapsed_time)
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    stream.stop()
    stream.close()

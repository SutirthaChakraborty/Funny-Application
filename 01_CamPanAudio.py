
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf

# Global variables
current_frame = 0
balance = 0


def scale_balance(balance):
    # Original range: [-10, 10]
    # New range: [0, 1]
    new_balance = (balance + 7) / 14
    return new_balance

# Audio callback function
def callback(outdata, frames, time, status):
    global current_frame
    global balance

    if current_frame + frames > len(left_wave):
        raise sd.CallbackStop

    left_data = left_wave[current_frame:current_frame + frames, :]
    right_data = right_wave[current_frame:current_frame + frames, :]

    if balance >=0:
        left_volume = max(min(0.5 + balance / 2, 1), 0)
        right_volume = max(min(0.5 - balance / 2, 1), 0)
    else:
        right_volume = max(min(0.5 + abs(balance) / 2, 1), 0)
        left_volume = max(min(0.5 - abs(balance) / 2, 1), 0)
    
    # print(left_volume,right_volume)

    outdata[:, 0] = left_data[:, 0] * left_volume
    outdata[:, 1] = right_data[:, 0] * right_volume

    current_frame += frames
    if current_frame >= len(left_wave):
        current_frame = 0

# Load wav files
left_wav_file = 'audio1.wav'
right_wav_file = 'audio1.wav'
# Load second video
cap2 = cv2.VideoCapture('vid.mp4')
prev_cropped_image2=np.zeros((100,100,3),np.uint8)
window_size = 100  # size of the moving window


left_wave, left_sr = sf.read(left_wav_file, dtype='float32', always_2d=True)
right_wave, right_sr = sf.read(right_wav_file, dtype='float32', always_2d=True)

# Initialize audio stream
blocksize = 1024

# stream = sd.OutputStream(samplerate=left_sr, channels=2, dtype='float32', blocksize=blocksize, callback=callback)
# stream.start()

def start_audio_stream():
    global stream
    stream = sd.OutputStream(samplerate=left_sr, channels=2, dtype='float32', blocksize=blocksize, callback=callback)
    stream.start()



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
# Ensure that the video has started before starting the audio
if cap.isOpened():
    start_audio_stream()


while cap.isOpened():
    success, image = cap.read()
    success2, image2 = cap2.read()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = round(angles[0] * 360)
            y = round(angles[1] * 360)

            # # See where the user's head tilting
            # if y < -10:
            #     text = "Looking Left"
            # elif y > 10:
            #     text = "Looking Right"
            # elif x < -10:
            #     text = "Looking Down"
            # else:
            #     text = "Forward"

            # Display the nose direction
            # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            # p1 = (int(nose_2d[0]), int(nose_2d[1]))
            # p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            # cv2.line(image, p1, p2, (255, 0, 0), 2)
                        
            balance=min(y/10,1)
            # Add the text on the image
            cv2.putText(image, str(y), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

            
        if success2:
            # Normalize the y value to the height of the second video
            dim = (image2.shape[1] - 100, image2.shape[0] - 100)
            width, height = image2.shape[1], image2.shape[0]
            crop_width = dim[0] if dim[0]<image2.shape[1] else image2.shape[1]
            crop_height = dim[1] if dim[1]<image2.shape[0] else image2.shape[0] 
            mid_x, mid_y = int(width/2), int(height/2)
            cw2, ch2 = int(crop_width/2), int(crop_height/2)
            x_factor= 5
            y_factor= 12
            
            a= mid_y-ch2+(x*5) if mid_y-ch2+( x * x_factor ) > 0 else 0
            b= mid_y+ch2+(x*5) if mid_y+ch2+( x * x_factor ) < image2.shape[0] else image2.shape[0]
            c= mid_x-cw2+(y*10)  if mid_x-cw2+( y * y_factor ) > 0 else 0
            d= mid_x+cw2+(y*10) if mid_x+cw2+( y * y_factor ) < image2.shape[1] else image2.shape[1]
            
            cropped_image2 = image2[a:b, c:d]
            cropped_image2 = cv2.resize(cropped_image2, dim, interpolation=cv2.INTER_AREA)

            # Display the cropped video 
            cv2.imshow('Cropped Video', cropped_image2)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()

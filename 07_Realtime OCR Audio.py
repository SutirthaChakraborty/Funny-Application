from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import time
import cv2
import pytesseract
from difflib import SequenceMatcher
import re
import simpleaudio as sa

import pyttsx3
engine = pyttsx3.init()
newVoiceRate = 145
engine.setProperty('rate',newVoiceRate)
# setting up tesseract path
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"  # For ARM-based Macs

# Default configuration values
VIDEO_PATH = None
EAST_MODEL_PATH = "frozen_east_text_detection.pb"
MIN_CONFIDENCE = 0.5
RESIZED_WIDTH = 320
RESIZED_HEIGHT = 320
PADDING = 0.0

import speech_recognition as sr

def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak your query:")
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio,language='en-in')
        print("You said: ", query)
        return query
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))



def box_extractor(scores, geometry, min_confidence):
    num_rows, num_cols = scores.shape[2:4]
    rectangles = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < min_confidence:
                continue

            offset_x, offset_y = x * 4.0, y * 4.0

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            box_h = x_data0[x] + x_data2[x]
            box_w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y + (cos * x_data2[x]) - (sin * x_data1[x]))
            start_x = int(end_x - box_w)
            start_y = int(end_y - box_h)

            rectangles.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rectangles, confidences


def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def play_frequency(freq):
    sample_rate = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(freq * t * 2 * np.pi)
    audio_data = (audio_data * 32767 / np.max(np.abs(audio_data))).astype(np.int16)
    play_obj = sa.play_buffer(audio_data, 1, 2, sample_rate)
    play_obj.wait_done()


if __name__ == '__main__':
    w, h = None, None
    new_w, new_h = RESIZED_WIDTH, RESIZED_HEIGHT
    ratio_w, ratio_h = None, None

    layer_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(EAST_MODEL_PATH)

    if VIDEO_PATH is None:
        print("[INFO] starting video stream...")
        vs = VideoStream(src=1).start()
        time.sleep(1)

    else:
        vs = cv2.VideoCapture(VIDEO_PATH)

    fps = FPS().start()
    engine.say("Ask for the direction you are looking for?")
    engine.runAndWait()

    # target_word = record_audio()
    target_word=input("Enter the word :: ")

    while True:
        frame = vs.read()
        frame = frame[1] if VIDEO_PATH is not None else frame

        if frame is None:
            break

        frame = imutils.resize(frame, width=1000)
        orig = frame.copy()
        orig_h, orig_w = orig.shape[:2]

        if w is None or h is None:
            h, w = frame.shape[:2]
            ratio_w = w / float(new_w)
            ratio_h = h / float(new_h)

        frame = cv2.resize(frame, (new_w, new_h))

        blob = cv2.dnn.blobFromImage(frame, 1.0, (new_w, new_h), (123.68, 116.78, 103.94),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        scores, geometry = net.forward(layer_names)

        rectangles, confidences = box_extractor(scores, geometry, min_confidence=MIN_CONFIDENCE)
        boxes = non_max_suppression(np.array(rectangles), probs=confidences)

        for (start_x, start_y, end_x, end_y) in boxes:

            start_x = int(start_x * ratio_w)
            start_y = int(start_y * ratio_h)
            end_x = int(end_x * ratio_w)
            end_y = int(end_y * ratio_h)

            dx = int((end_x - start_x) * PADDING)
            dy = int((end_y - start_y) * PADDING)

            start_x = max(0, start_x - dx)
            start_y = max(0, start_y - dy)
            end_x = min(orig_w, end_x + (dx * 2))
            end_y = min(orig_h, end_y + (dy * 2))

            # ROI to be recognized
            roi = orig[start_y:end_y, start_x:end_x]

            # recognizing text
            config = '-l eng --oem 1 --psm 7'
            raw_text = pytesseract.image_to_string(roi, config=config)
            text = re.sub('[^a-zA-Z0-9]', ' ', raw_text)

            similarity_threshold = 0.6
            similar_word = ""
            max_similarity = 0

            for word in text.split():
                similarity = string_similarity(target_word, word)
                if similarity > max_similarity:
                    max_similarity = similarity
                    similar_word = word

            if max_similarity > similarity_threshold:
                center_x = (start_x + end_x) // 2
                freq = 200 + (center_x / orig_w) * 800
                play_frequency(freq)

                cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                cv2.putText(orig, text, (start_x, start_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        fps.update()

        cv2.imshow("Detection", orig)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    fps.stop()
    print(f"[INFO] elapsed time {round(fps.elapsed(), 2)}")
    print(f"[INFO] approx. FPS : {round(fps.fps(), 2)}")

    if VIDEO_PATH is None:
        vs.stop()

    else:
        vs.release()

    cv2.destroyAllWindows()

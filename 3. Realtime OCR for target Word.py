import cv2
import pytesseract
import difflib
from pytesseract import Output

# Set the tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"  # For ARM-based Macs

def display_text_coordinates(image, words_data):
    for word_data in words_data:
        x, y, w, h, text = word_data
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def detect_text_coordinates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, output_type=Output.DICT, config=custom_config)
    
    words_data = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:
            x, y, w, h, text = data['left'][i], data['top'][i], data['width'][i], data['height'][i], data['text'][i]
            words_data.append((x, y, w, h, text))
    return words_data

def word_similarity(a, b):
    seq_matcher = difflib.SequenceMatcher(None, a, b)
    return seq_matcher.ratio()

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    target_word = input("Enter the word you want to search for: ")
    # similarity_threshold = float(input("Enter the similarity threshold (0.0 to 1.0): "))
    similarity_threshold=0.5
    frame_skip_rate = 7

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip_rate == 0:
            downscaled_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            words_data = detect_text_coordinates(downscaled_frame)

            for word_data in words_data:
                x, y, w, h, text = word_data
                similarity = word_similarity(text.lower(), target_word.lower())
                if similarity >= similarity_threshold:
                    x, y, w, h = x * 2, y * 2, w * 2, h * 2
                    display_text_coordinates(frame, [(x, y, w, h, text)])
                    break

        cv2.imshow('Webcam OCR', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

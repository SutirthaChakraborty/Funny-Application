import sys
import pyautogui
import numpy as np
import easyocr
import g4f
import threading
from flask import Flask, render_template_string, request
import logging
from pynput import keyboard

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Flask app for hosting the webpage
app = Flask(__name__)
latest_response = "No data yet."


@app.route('/')
def index():
    return render_template_string(
        '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>Screen Assistant Output</title>
      </head>
      <body>
        <h1>Screen Assistant Output</h1>
        <div id="output">{{ latest_response }}</div>
        <script>
          setInterval(function() {
            fetch('/get_latest').then(response => response.text()).then(data => {
              document.getElementById('output').innerText = data;
            });
          }, 2000);
        </script>
      </body>
    </html>
    ''',
        latest_response=latest_response,
    )


@app.route('/get_latest')
def get_latest():
    return latest_response


def run_flask_app():
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)


# Start Flask in a separate thread
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.daemon = True
flask_thread.start()


def process_screen():
    global latest_response
    try:
        # Capture screenshot
        screenshot = pyautogui.screenshot()

        # Perform OCR
        image_np = np.array(screenshot)
        results = reader.readtext(image_np)
        extracted_text = " ".join(result[1] for result in results)

        if extracted_text.strip():
            # Get GPT response
            response = g4f.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "assistant",
                        "content": f"Answer question/ solve error to this text from OCR of screenshot: {extracted_text}. DO NOT write extra commentary. Just Answer to the point",
                    }
                ],
            )
            response_text = "".join(response)
            latest_response = response_text
            print(f"Extracted Text: {extracted_text}\nGPT Response: {response_text}")
        else:
            latest_response = "No text detected on screen."
            print("No text detected on screen.")
    except Exception as e:
        latest_response = f"Error processing screen: {str(e)}"
        print(f"Error processing screen: {str(e)}")


# Set up global hotkey using pynput
logging.basicConfig(level=logging.DEBUG)
logging.debug("Hotkey listener set.")


def on_activate():
    logging.debug("Hotkey activated, starting screen processing.")
    process_screen()


def for_canonical(f):
    return lambda k: f(listener.canonical(k))


hotkey = keyboard.HotKey(keyboard.HotKey.parse('<ctrl>+<shift>+z'), on_activate)

listener = keyboard.Listener(
    on_press=for_canonical(hotkey.press), on_release=for_canonical(hotkey.release)
)
listener.start()

if __name__ == '__main__':
    listener.join()

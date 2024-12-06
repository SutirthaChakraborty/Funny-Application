from flask import Flask, request, render_template_string, jsonify
import pyautogui
import os
import threading
import asyncio
import logging
from pynput import keyboard
import g4f
from g4f.client import AsyncClient

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)

# Directory to save screenshots
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Global variables
conversation_history = []
latest_response = "No data yet."

# Initialize the g4f client
gpt_client = AsyncClient(provider=g4f.Provider.Blackbox)


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
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            #output {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                min-height: 100px;
            }
            .timestamp {
                color: #666;
                font-size: 0.8em;
            }
        </style>
      </head>
      <body>
        <h1>Screen Assistant Output</h1>
        <div id="output">{{ latest_response }}</div>
        <div class="timestamp">Last updated: <span id="timestamp">Never</span></div>
        <script>
          setInterval(function() {
            fetch('/get_latest')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('output').innerText = data.response;
                    document.getElementById('timestamp').innerText = new Date().toLocaleTimeString();
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
    return jsonify({"response": latest_response})


async def analyze_image_and_continue_chat(image_path):
    global latest_response
    try:
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()

        # Create the chat completion with the image
        response = await gpt_client.chat.completions.create(
            model=g4f.models.default,
            messages=[
                {
                    "role": "user",
                    "content": "Understand the image,if there is a question solve it? Write in plain text. DONOT WRITE EXTRA COMMENTARY. No Bold No italics",
                }
            ],
            image=image_data,
        )

        gpt_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": gpt_response})
        latest_response = gpt_response
        logger.debug(f"Received response: {gpt_response}")
        return gpt_response

    except Exception as e:
        error_message = f"An error occurred during analysis: {e}"
        latest_response = error_message
        logger.error(error_message)
        return error_message


async def process_screen_async():
    """Process the screen asynchronously when hotkey is activated"""
    logger.debug("Processing screen...")
    try:
        screenshot_path = os.path.join(SCREENSHOT_DIR, 'screenshot.png')
        screenshot = pyautogui.screenshot()
        screenshot.save(screenshot_path)

        response = await analyze_image_and_continue_chat(screenshot_path)
        logger.debug(f"Analysis complete: {response}")
    except Exception as e:
        logger.error(f"Error processing screen: {e}")


def on_activate():
    """Callback for when the hotkey is pressed"""
    logger.debug("Hotkey activated, starting screen processing.")
    asyncio.run(process_screen_async())


def setup_hotkey():
    """Set up the global hotkey listener"""
    with keyboard.GlobalHotKeys({'z': on_activate}) as h:
        h.join()


def run_flask_app():
    """Run the Flask web application"""
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)


if __name__ == '__main__':
    try:
        # Start Flask in a separate thread
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()
        logger.info("Flask application started")

        # Start the hotkey listener in the main thread
        logger.info("Starting hotkey listener (Z)")
        setup_hotkey()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

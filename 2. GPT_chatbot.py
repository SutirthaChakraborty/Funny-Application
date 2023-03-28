# Listen's to you and answer's your questions.

import pyaudio
import speech_recognition as sr
import openai
import threading
import queue
from playsound import playsound
from gtts import gTTS
import os
import tempfile

# Set up OpenAI API key (replace 'your_openai_api_key' with your actual API key)
openai.api_key = 'API-KEY'

speak_queue = queue.Queue()

def speak():
    while True:
        text = speak_queue.get()
        if text is None:
            break
        tts = gTTS(text, lang='en')
        with tempfile.NamedTemporaryFile(delete=True) as fp:
            temp_path = fp.name
        tts.save(temp_path)
        playsound(temp_path)

speak_thread = threading.Thread(target=speak)
speak_thread.start()

# Record audio from the microphone
def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak your query:")
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio, language='en-in')
        print("You said: ", query)
        return query
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def chatting(query):
    response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    n=1,
    messages=[
        {"role": "system", "content": "You are a helpful assistant with exciting, interesting things to say. Just Answer questions as asked to the point"},
        {"role": "user", "content": query},
    ])

    message = response.choices[0]['message']
    print("{}: {}".format(message['role'], message['content']))
    return message['content']

if __name__ == "__main__":
    while True:
        query = record_audio()
        if query:
            gpt_response = chatting(query)
            
            # Add the response to the speak queue
            speak_queue.put(gpt_response)

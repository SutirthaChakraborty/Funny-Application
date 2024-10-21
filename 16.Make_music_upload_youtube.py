# Import necessary libraries
import os
import shutil
import subprocess
import json
import tempfile
import torch
import torchaudio
import soundfile as sf
from yt_dlp import YoutubeDL
from audiocraft.models import MusicGen, MultiBandDiffusion
import demucs.separate
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import requests
from bs4 import BeautifulSoup


# Function to download YouTube audio using yt-dlp
def download_audio_from_youtube(link, output_dir="downloads"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }
        ],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),  # Save with title
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(link, download=True)
            output_filename = ydl.prepare_filename(info_dict)
            wav_filename = (
                output_filename.rsplit('.', 1)[0] + '.wav'
            )  # Ensure it's .wav
            return wav_filename, info_dict  # Return both file path and metadata
    except Exception as e:
        print(f"Error downloading or converting audio: {e}")
        return None, None


# Function to remove drums using Demucs (optional based on user input)
def remove_drums(
    audio_path,
    output_dir="/data/schakraborty/user_interaction/separated",
    remove_drums=False,
):
    if not remove_drums:
        return audio_path  # If not removing drums, return the original path

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Call the Demucs separation function directly with arguments
        demucs.separate.main(
            [
                "--mp3",  # Output as MP3
                "--two-stems",
                "drums",  # Only separate drums
                "-n",
                "mdx_extra",  # Use the 'mdx_extra' model
                audio_path,  # Input track
            ]
        )

        # Construct the expected output path for the no-drums file
        track_name = os.path.splitext(os.path.basename(audio_path))[
            0
        ]  # Extract track name without extension
        no_drums_path = os.path.join(
            output_dir, "mdx_extra", track_name, "no_drums.mp3"
        )

        # Ensure that the no-drum version exists
        if os.path.exists(no_drums_path):
            return no_drums_path
        else:
            print(f"No drums file could be found at: {no_drums_path}")
            return None
    except Exception as e:
        print(f"Error during drum removal: {e}")
        return None


# Function to apply a fade-out effect
def apply_fade_out(waveform, sr, duration=5):
    fade_length = int(sr * duration)
    fade_curve = torch.linspace(1, 0, fade_length).to(waveform.device)

    # Apply fade-out at the end of the waveform
    waveform[:, -fade_length:] *= fade_curve
    return waveform


# Function to generate similar audio using melody-conditional generation
def generate_similar_audio_with_melody(output_path, duration=200):
    try:
        # Load the pre-trained MusicGen model for melody conditioning
        model = MusicGen.get_pretrained('facebook/musicgen-stereo-melody-large')
        model.set_generation_params(duration=duration)

        # Load the YouTube audio (melody without drums)
        melody_waveform, sr = torchaudio.load(output_path)

        # Convert mono to stereo if necessary
        if melody_waveform.size(0) == 1:
            melody_waveform = melody_waveform.repeat(2, 1)
        elif melody_waveform.size(0) > 2:
            melody_waveform = melody_waveform[:2, :]  # Limit to stereo

        device = (
            torch.device('mps')
            if torch.backends.mps.is_available()
            else (
                torch.device('cuda')
                if torch.cuda.is_available()
                else torch.device('cpu')
            )
        )
        melody_waveform = melody_waveform.to(device)

        # Provide descriptions for generating similar music. Modify this description as needed.
        description = ["enigma style, Shakuhachi, romantic, 3D, HD, peace"]

        generated_music = model.generate_with_chroma(
            descriptions=description,
            melody_wavs=melody_waveform.unsqueeze(0),
            melody_sample_rate=sr,
            progress=True,
        )

        # Apply fade-out effect to the generated music
        generated_music_faded = apply_fade_out(
            generated_music[0].cpu(), 32000, duration=5
        )

        # Save the output as a .wav file
        generated_output_path = os.path.join(
            os.path.dirname(output_path), "generated_similar_with_melody_fadeout.wav"
        )
        torchaudio.save(generated_output_path, generated_music_faded, sample_rate=32000)
        print(f"Generated audio with fade-out saved to {generated_output_path}")
        return generated_output_path

    except Exception as e:
        print(f"Error during generation: {e}")
        return None


# Function to add audio to video and generate output video
def add_audio_to_video(video_path, audio_path, output_path):
    # Load the video and audio files
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # Determine the duration of video and audio
    video_duration = video.duration
    audio_duration = audio.duration

    # Adjust the video length to match the audio length
    if audio_duration < video_duration:
        # Trim the video to match the audio length
        video = video.subclip(0, audio_duration)
    elif audio_duration > video_duration:
        # Loop the video to match the audio length
        clips = []
        remaining_duration = audio_duration
        while remaining_duration > 0:
            clip_duration = min(video_duration, remaining_duration)
            clips.append(video.subclip(0, clip_duration))
            remaining_duration -= clip_duration
        video = concatenate_videoclips(clips)

    # Set the audio to the video
    video = video.set_audio(audio)

    # Ensure fps is set to a valid value
    fps = video.fps if video.fps and video.fps > 0 else 24

    # Ensure video size is properly set
    if not video.size:
        raise ValueError("Video size is not properly set. Please check the video file.")

    # Write the output video file
    video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=fps)


# Function to get YouTube keywords
def get_youtube_keywords(url):
    # Send a request to fetch the HTML content of the video page
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Search for the meta tag containing keywords
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})

        # Extract the content attribute from the meta tag
        if keywords_meta:
            keywords = keywords_meta.get('content', None)
            if keywords:
                return keywords.split(', ')
        else:
            return "Keywords meta tag not found."
    else:
        return f"Failed to retrieve the page. Status code: {response.status_code}"


# Main function to run the entire pipeline
def main():
    youtube_link = input("Enter the YouTube link: ")
    duration = int(input("Enter the duration in seconds: "))
    remove_drum_option = (
        input("Do you want to remove drums? (yes/no): ").strip().lower() == 'yes'
    )

    # Step 1: Download the audio using yt-dlp
    audio_path, info_dict = download_audio_from_youtube(youtube_link)
    if audio_path:
        print(f"Audio downloaded and saved to: {audio_path}")

        # Step 2: Optionally remove drums
        no_drums_audio_path = remove_drums(audio_path, remove_drums=remove_drum_option)
        if no_drums_audio_path:
            print(f"Drums removed and saved to: {no_drums_audio_path}")

            # Step 3: Generate similar audio using melody-conditional generation
            generated_audio_path = generate_similar_audio_with_melody(
                no_drums_audio_path, duration=duration
            )
            if generated_audio_path:
                print("Audio processing pipeline completed!")

                # Step 4: Add audio to video and generate output video
                video_path = "Visualizer.mp4"
                output_video_path = os.path.join(
                    os.path.dirname(generated_audio_path), "output_video.mp4"
                )
                add_audio_to_video(video_path, generated_audio_path, output_video_path)
                print(f"Video generated and saved to: {output_video_path}")

                # Step 5: Upload to YouTube using youtube.py script
                keywords = get_youtube_keywords(youtube_link)
                if isinstance(keywords, list):
                    keywords_str = ", ".join(keywords)
                else:
                    keywords_str = "sutirtha, best, mobile, upload"

                command = [
                    "python3",
                    "youtube.py",
                    "--file",
                    output_video_path,
                    "--title",
                    f"{info_dict['title']} | Sutirtha | Soulz Muzik | Remix",
                    "--description",
                    info_dict.get('description', ''),
                    "--keywords",
                    keywords_str,
                    "--category",
                    "22",
                    "--privacyStatus",
                    "private",
                ]

                subprocess.call(command)
            else:
                print("Failed to generate similar audio.")
        else:
            print("Failed to process drums.")
    else:
        print("Failed to download the audio.")


#  ---------------------------- youtube.py
#!/usr/bin/python

import httplib2
import os
import random
import sys
import time

from apiclient.discovery import build
from apiclient.errors import HttpError
from apiclient.http import MediaFileUpload
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow


# Explicitly tell the underlying HTTP transport library not to retry, since
# we are handling retry logic ourselves.
httplib2.RETRIES = 1

# Maximum number of times to retry before giving up.
MAX_RETRIES = 10

# Always retry when these exceptions are raised.
RETRIABLE_EXCEPTIONS = (httplib2.HttpLib2Error, IOError)

# Always retry when an apiclient.errors.HttpError with one of these status
# codes is raised.
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret. You can acquire an OAuth 2.0 client ID and client secret from
# the Google API Console at
# https://console.cloud.google.com/.
# Please ensure that you have enabled the YouTube Data API for your project.
# For more information about using OAuth2 to access the YouTube Data API, see:
#   https://developers.google.com/youtube/v3/guides/authentication
# For more information about the client_secrets.json file format, see:
#   https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
CLIENT_SECRETS_FILE = "client_secrets.json"

# This OAuth 2.0 access scope allows an application to upload files to the
# authenticated user's YouTube channel, but doesn't allow other types of access.
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# This variable defines a message to display if the CLIENT_SECRETS_FILE is
# missing.
MISSING_CLIENT_SECRETS_MESSAGE = """
WARNING: Please configure OAuth 2.0

To make this sample run you will need to populate the client_secrets.json file
found at:

   %s

with information from the API Console
https://console.cloud.google.com/

For more information about the client_secrets.json file format, please visit:
https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
""" % os.path.abspath(
    os.path.join(os.path.dirname(__file__), CLIENT_SECRETS_FILE)
)

VALID_PRIVACY_STATUSES = ("public", "private", "unlisted")


def get_authenticated_service(args):
    flow = flow_from_clientsecrets(
        CLIENT_SECRETS_FILE,
        scope=YOUTUBE_UPLOAD_SCOPE,
        message=MISSING_CLIENT_SECRETS_MESSAGE,
    )

    storage = Storage("%s-oauth2.json" % sys.argv[0])
    credentials = storage.get()

    if credentials is None or credentials.invalid:
        credentials = run_flow(flow, storage, args)

    return build(
        YOUTUBE_API_SERVICE_NAME,
        YOUTUBE_API_VERSION,
        http=credentials.authorize(httplib2.Http()),
    )


def initialize_upload(youtube, options):
    tags = None
    if options.keywords:
        tags = options.keywords.split(",")

    body = dict(
        snippet=dict(
            title=options.title,
            description=options.description,
            tags=tags,
            categoryId=options.category,
        ),
        status=dict(privacyStatus=options.privacyStatus),
    )

    # Call the API's videos.insert method to create and upload the video.
    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        # The chunksize parameter specifies the size of each chunk of data, in
        # bytes, that will be uploaded at a time. Set a higher value for
        # reliable connections as fewer chunks lead to faster uploads. Set a lower
        # value for better recovery on less reliable connections.
        #
        # Setting "chunksize" equal to -1 in the code below means that the entire
        # file will be uploaded in a single HTTP request. (If the upload fails,
        # it will still be retried where it left off.) This is usually a best
        # practice, but if you're using Python older than 2.6 or if you're
        # running on App Engine, you should set the chunksize to something like
        # 1024 * 1024 (1 megabyte).
        media_body=MediaFileUpload(options.file, chunksize=-1, resumable=True),
    )

    resumable_upload(insert_request)


# This method implements an exponential backoff strategy to resume a
# failed upload.


def resumable_upload(insert_request):
    response = None
    error = None
    retry = 0
    while response is None:
        try:
            print("Uploading file...")
            status, response = insert_request.next_chunk()
            if response is not None:
                if 'id' in response:
                    print("Video id '%s' was successfully uploaded." % response['id'])
                else:
                    exit("The upload failed with an unexpected response: %s" % response)
        except HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                error = "A retriable HTTP error %d occurred:\n%s" % (
                    e.resp.status,
                    e.content,
                )
            else:
                raise
        except RETRIABLE_EXCEPTIONS as e:
            error = "A retriable error occurred: %s" % e

        if error is not None:
            print(error)
            retry += 1
            if retry > MAX_RETRIES:
                exit("No longer attempting to retry.")

            max_sleep = 2**retry
            sleep_seconds = random.random() * max_sleep
            print("Sleeping %f seconds and then retrying..." % sleep_seconds)
            time.sleep(sleep_seconds)


if __name__ == '__main__':
    argparser.add_argument("--file", required=True, help="Video file to upload")
    argparser.add_argument("--title", help="Video title", default="Test Title")
    argparser.add_argument(
        "--description", help="Video description", default="Test Description"
    )
    argparser.add_argument(
        "--category",
        default="22",
        help="Numeric video category. "
        + "See https://developers.google.com/youtube/v3/docs/videoCategories/list",
    )
    argparser.add_argument(
        "--keywords", help="Video keywords, comma separated", default=""
    )
    argparser.add_argument(
        "--privacyStatus",
        choices=VALID_PRIVACY_STATUSES,
        default=VALID_PRIVACY_STATUSES[0],
        help="Video privacy status.",
    )
    args = argparser.parse_args()

    if not os.path.exists(args.file):
        exit("Please specify a valid file using the --file= parameter.")

    youtube = get_authenticated_service(args)
    try:
        initialize_upload(youtube, args)
    except HttpError as e:
        print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))






if __name__ == "__main__":
    main()

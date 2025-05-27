import gradio as gr
from pydub import AudioSegment
from pydub.playback import play
from audiostretchy.stretch import stretch_audio
import tempfile
import numpy as np


# Function to change the speed using AudioStretchy (faster if speed > 1, slower if speed < 1)
def change_speed_without_pitch(file, speed=1.0):
    output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    # Invert the speed to match the correct faster/slower logic for AudioStretchy
    stretch_ratio = 1.0 / speed  # 2x faster means a ratio of 0.5
    stretch_audio(file, output_file, ratio=stretch_ratio)

    return output_file


# Function to change the pitch using pydub
def change_pitch_with_pydub(file, semitones=0):
    sound = AudioSegment.from_file(file)

    # Shift the pitch by changing the frame rate
    new_sample_rate = int(sound.frame_rate * (2.0 ** (semitones / 12.0)))

    # Change the frame rate but keep the same sample rate
    pitch_shifted_sound = sound._spawn(
        sound.raw_data, overrides={'frame_rate': new_sample_rate}
    )
    pitch_shifted_sound = pitch_shifted_sound.set_frame_rate(sound.frame_rate)

    # Save to a new temp file
    output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    pitch_shifted_sound.export(output_file, format="wav")

    return output_file


# Main function to apply both speed and pitch changes
def process_audio(file, speed, pitch):
    # First, adjust the speed without changing the pitch
    speed_adjusted_file = change_speed_without_pitch(file, speed=speed)

    # Then, apply pitch shifting to the speed-adjusted file
    final_file = change_pitch_with_pydub(speed_adjusted_file, semitones=pitch)

    return final_file


# Gradio interface
interface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(type="filepath", label="Upload your audio"),
        gr.Slider(
            minimum=0.25,
            maximum=4.0,
            step=0.1,
            value=1.0,
            label="Speed (0.25x to 4.0x)",
        ),
        gr.Slider(
            minimum=-12, maximum=12, step=1, value=0, label="Pitch (in Semitones)"
        ),
    ],
    outputs=gr.Audio(label="Modified Audio"),
    title="Audio Speed and Pitch Changer",
    description="Upload an audio file, adjust the speed (faster or slower), and change the pitch.",
)

# Launch the Gradio interface
interface.launch()


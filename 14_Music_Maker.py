# audiocraft==1.3.0
# audiosr==0.0.7
# demucs==4.0.1
# pydub==0.25.1
# soundfile==0.12.1
# torch==2.1.0
# torchaudio==2.1.0
# yt_dlp==2024.10.7


import os
import shutil
import subprocess
import json
import tempfile
import torch
import torchaudio
import soundfile as sf
from yt_dlp import YoutubeDL
from audiocraft.models import MusicGen
import demucs.separate
from pydub import AudioSegment
from audiosr import build_model, super_resolution

# Function to download YouTube audio using yt-dlp
def download_audio_from_youtube(link, output_dir="downloads"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),  # Save with title
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(link, download=True)
            output_filename = ydl.prepare_filename(info_dict)
            wav_filename = output_filename.rsplit('.', 1)[0] + '.wav'  # Ensure it's .wav
            return wav_filename
    except Exception as e:
        print(f"Error downloading or converting audio: {e}")
        return None

# Function to remove drums using Demucs (optional based on user input)
def remove_drums(audio_path, output_dir="/data/schakraborty/user_interaction/separated", remove_drums=False):
    if not remove_drums:
        return audio_path  # If not removing drums, return the original path

    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Call the Demucs separation function directly with arguments
        demucs.separate.main([
            "--mp3",                   # Output as MP3
            "--two-stems", "drums",      # Only separate drums
            "-n", "mdx_extra",          # Use the 'mdx_extra' model
            audio_path                 # Input track
        ])

        # Construct the expected output path for the no-drums file
        track_name = os.path.splitext(os.path.basename(audio_path))[0]  # Extract track name without extension
        no_drums_path = os.path.join(output_dir, "mdx_extra", track_name, "no_drums.mp3")

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
        model = MusicGen.get_pretrained('facebook/musicgen-melody-large')
        model.set_generation_params(duration=duration)

        # Load the YouTube audio (melody without drums)
        melody_waveform, sr = torchaudio.load(output_path)

        # Convert mono to stereo if necessary
        if melody_waveform.size(0) == 1:
            melody_waveform = melody_waveform.repeat(2, 1)
        elif melody_waveform.size(0) > 2:
            melody_waveform = melody_waveform[:2, :]  # Limit to stereo

        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        melody_waveform = melody_waveform.to(device)

        # Provide descriptions for generating similar music. Modify this description as needed.
        description = [
            "enigma style, Shakuhachi, romantic, 3D, HD, peace"
        ]

        generated_music = model.generate_with_chroma(
            descriptions=description,
            melody_wavs=melody_waveform.unsqueeze(0),
            melody_sample_rate=sr,
            progress=True
        )

        # Apply fade-out effect to the generated music
        generated_music_faded = apply_fade_out(generated_music[0].cpu(), 32000, duration=5)

        # Save the output as a .wav file
        generated_output_path = os.path.join(os.path.dirname(output_path), "generated_similar_with_melody_fadeout.wav")
        torchaudio.save(generated_output_path, generated_music_faded, sample_rate=32000)
        print(f"Generated audio with fade-out saved to {generated_output_path}")
        return generated_output_path

    except Exception as e:
        print(f"Error during generation: {e}")
        return None

# Function to get the sample rate of an audio file
def get_sample_rate(file_path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate', '-of', 'json', file_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = json.loads(result.stdout)
    return int(output['streams'][0]['sample_rate'])

# Function to split audio into chunks
def split_audio(input_path, chunk_duration_ms=5120):
    audio = AudioSegment.from_file(input_path)
    total_duration_ms = len(audio)
    num_chunks = total_duration_ms // chunk_duration_ms + 1
    chunk_files = []

    for i in range(num_chunks):
        start_time = i * chunk_duration_ms
        end_time = min((i + 1) * chunk_duration_ms, total_duration_ms)
        chunk = audio[start_time:end_time]
        chunk_filename = os.path.join(tempfile.gettempdir(), f"chunk_{i}.wav")
        chunk.export(chunk_filename, format="wav")
        chunk_files.append(chunk_filename)

    return chunk_files


# Function to merge processed audio chunks in proper sequence
def merge_audio_chunks(chunk_folder, output_file):
    final_audio = AudioSegment.empty()

    # Extract numerical index from chunk filenames and sort them
    def chunk_key(chunk_file):
        # Extract the index number from the filename (e.g., chunk_2.wav -> 2)
        base_name = os.path.basename(chunk_file)
        index_str = base_name.split('_')[1].split('.')[0]
        return int(index_str)  # Convert the extracted string to an integer

    # List all chunk files in the folder and sort them by their numeric index
    chunks = sorted([os.path.join(chunk_folder, f) for f in os.listdir(chunk_folder) if f.endswith('.wav')], key=chunk_key)

    # Merge the chunks in order
    for chunk_file in chunks:
        chunk = AudioSegment.from_file(chunk_file)
        final_audio += chunk

    # Export the final merged audio
    final_audio.export(output_file, format="wav")
    print(f"Merged audio saved to {output_file}")

# New function to save audio
def save_audio(waveform, savepath, name="outwav", samplerate=16000):
    if type(name) is not list:
        name = [name] * waveform.shape[0]
    for i in range(waveform.shape[0]):
        fname = f"{os.path.basename(name[i])}.wav"
        path = os.path.join(savepath, fname)
        sf.write(path, waveform[i, 0], samplerate=samplerate)
        print(f"Saving audio to {path}")

# Function to process audio using AudioSR
def process_audio_with_audiosr(input_audio, audiosr, chunk_duration_ms=5120):
    chunks = split_audio(input_audio, chunk_duration_ms)
    processed_chunks_dir = tempfile.mkdtemp(prefix="processed_chunks_")
    for chunk in chunks:
        try:
            waveform = super_resolution(audiosr, chunk, seed=42, guidance_scale=3.5, ddim_steps=50, latent_t_per_second=12.8)
            base_name, _ = os.path.splitext(os.path.basename(chunk))
            output_chunk_path = os.path.join(processed_chunks_dir, f"{base_name}_processed.wav")
            save_audio(waveform, savepath=processed_chunks_dir, name=base_name, samplerate=48000)
        except Exception as e:
            print(f"An error occurred while processing chunk {chunk}: {e}")
    return processed_chunks_dir

# Main function to run the entire pipeline
def main():
    youtube_link = input("Enter the YouTube link: ")
    duration = int(input("Enter the duration in seconds: "))
    remove_drum_option = input("Do you want to remove drums? (yes/no): ").strip().lower() == 'yes'

    # Step 1: Download the audio using yt-dlp
    audio_path = download_audio_from_youtube(youtube_link)
    if audio_path:
        print(f"Audio downloaded and saved to: {audio_path}")

        # Step 2: Optionally remove drums
        no_drums_audio_path = remove_drums(audio_path, remove_drums=remove_drum_option)
        if no_drums_audio_path:
            print(f"Drums removed and saved to: {no_drums_audio_path}")

            # Step 3: Generate similar audio using melody-conditional generation
            generated_audio_path = generate_similar_audio_with_melody(no_drums_audio_path, duration=duration)
            if generated_audio_path:
                # Step 4: Apply super-resolution to the generated audio
                audiosr = build_model(model_name='basic', device="auto")
                print("Processing audio in chunks...")
                processed_chunks_dir = process_audio_with_audiosr(generated_audio_path, audiosr)

                # Step 5: Merge processed chunks into one audio file
                output_audio = os.path.join(os.path.dirname(generated_audio_path), "HD_" + os.path.basename(generated_audio_path))
                merge_audio_chunks(processed_chunks_dir, output_audio)

                print("Audio processing pipeline completed!")
            else:
                print("Failed to generate similar audio.")
        else:
            print("Failed to process drums.")
    else:
        print("Failed to download the audio.")

if __name__ == "__main__":
    main()

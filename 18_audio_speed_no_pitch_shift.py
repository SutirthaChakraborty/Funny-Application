import argparse
from pydub import AudioSegment
import os
import numpy as np
from audiomentations import TimeStretch
import io
import librosa


def change_speed_preserve_pitch(file, speed=1.0):
    """
    Change speed without changing pitch using audiomentations library.
    This provides professional-grade audio processing with better pitch preservation.
    """
    print(f"Changing speed to {speed}x while preserving original pitch using audiomentations...")
    
    # Load audio file using pydub
    sound = AudioSegment.from_file(file)
    original_duration = len(sound)
    
    # Convert to numpy array for audiomentations
    # First convert to mono and get raw audio data
    mono_sound = sound.set_channels(1)
    samples = np.array(mono_sound.get_array_of_samples(), dtype=np.float32)
    
    # Normalize to [-1, 1] range as expected by audiomentations
    samples = samples / (2**15)  # 16-bit audio normalization
    
    # Apply time stretch while preserving pitch
    time_stretch = TimeStretch(min_rate=speed, max_rate=speed, p=1.0)
    augmented_samples = time_stretch(samples=samples, sample_rate=sound.frame_rate)
    
    # Convert back to pydub format
    # Denormalize and convert to int16
    augmented_samples = (augmented_samples * (2**15)).astype(np.int16)
    
    # Create new AudioSegment from processed samples
    final_sound = AudioSegment(
        augmented_samples.tobytes(),
        frame_rate=sound.frame_rate,
        sample_width=sound.sample_width,
        channels=1
    )
    
    # If original was stereo, convert back to stereo by duplicating the channel
    if sound.channels == 2:
        final_sound = AudioSegment.from_mono_audiosegments(final_sound, final_sound)
    
    new_duration = len(final_sound)
    expected_duration = original_duration / speed
    
    print(f"Original duration: {original_duration/1000:.1f}s")
    print(f"New duration: {new_duration/1000:.1f}s")
    print(f"Expected duration: {expected_duration/1000:.1f}s")
    print(f"Using audiomentations TimeStretch with rate: {speed}")
    print(f"Pitch preservation: Professional-grade algorithm")
    
    return final_sound


def analyze_audio_features(file_path, label="Audio"):
    """
    Analyze audio features including beat timing and key/chord detection.
    """
    print(f"\n--- Analyzing {label} ---")
    
    # Load audio with librosa
    y, sr = librosa.load(file_path)
    
    # Get beat timing
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time')[1]
    
    print(f"Estimated tempo: {tempo:.2f} BPM")
    print(f"Number of beats detected: {len(beat_times)}")
    print(f"Song duration: {len(y)/sr:.2f} seconds")
    
    # Print first 10 beat timestamps
    print(f"First 10 beat timestamps (seconds):")
    for i, beat_time in enumerate(beat_times[:10]):
        print(f"  Beat {i+1}: {beat_time:.3f}s")
    
    if len(beat_times) > 10:
        print(f"  ... and {len(beat_times)-10} more beats")
    
    # Get chroma features for key/chord analysis
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Estimate key using chroma features
    # Calculate the mean chroma vector
    chroma_mean = np.mean(chroma, axis=1)
    
    # Define note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Find the most prominent note (rough key estimation)
    key_index = np.argmax(chroma_mean)
    estimated_key = note_names[key_index]
    
    print(f"Estimated key center: {estimated_key}")
    
    # Show chroma distribution (chord tendencies)
    print("Chroma distribution (note prominence):")
    for i, note in enumerate(note_names):
        prominence = chroma_mean[i]
        bar_length = int(prominence * 20)  # Scale for visualization
        bar = '█' * bar_length
        print(f"  {note:2s}: {prominence:.3f} {bar}")
    
    def detect_chord_quality(root_idx, chroma_vector, note_names):
        """
        Detect if a chord is major or minor based on the third interval.
        """
        # Major third is 4 semitones from root
        major_third_idx = (root_idx + 4) % 12
        # Minor third is 3 semitones from root
        minor_third_idx = (root_idx + 3) % 12
        # Perfect fifth is 7 semitones from root
        fifth_idx = (root_idx + 7) % 12
        
        major_third_strength = chroma_vector[major_third_idx]
        minor_third_strength = chroma_vector[minor_third_idx]
        fifth_strength = chroma_vector[fifth_idx]
        root_strength = chroma_vector[root_idx]
        
        # Check if we have a clear triad (root + third + fifth should be prominent)
        triad_threshold = 0.3
        if root_strength < triad_threshold:
            return "?"
        
        # Determine major vs minor based on third interval
        if major_third_strength > minor_third_strength:
            if major_third_strength > triad_threshold and fifth_strength > triad_threshold:
                return "maj"
            elif major_third_strength > triad_threshold:
                return "maj"
        elif minor_third_strength > major_third_strength:
            if minor_third_strength > triad_threshold and fifth_strength > triad_threshold:
                return "min"
            elif minor_third_strength > triad_threshold:
                return "min"
        
        # If unclear, return ambiguous
        return "?"
    
    # Analyze chord progressions in segments
    print("\nChord analysis (by segments):")
    segment_length = sr * 4  # 4-second segments
    num_segments = min(8, len(y) // segment_length)  # Analyze first 8 segments or less
    
    for i in range(num_segments):
        start_sample = i * segment_length
        end_sample = min((i + 1) * segment_length, len(y))
        segment = y[start_sample:end_sample]
        
        segment_chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        segment_chroma_mean = np.mean(segment_chroma, axis=1)
        
        # Find top 3 notes in this segment
        top_notes_indices = np.argsort(segment_chroma_mean)[-3:][::-1]
        
        # Analyze each top note as potential chord root
        chord_analysis = []
        for note_idx in top_notes_indices:
            note_name = note_names[note_idx]
            chord_quality = detect_chord_quality(note_idx, segment_chroma_mean, note_names)
            
            if chord_quality == "maj":
                chord_analysis.append(f"{note_name}maj")
            elif chord_quality == "min":
                chord_analysis.append(f"{note_name}min")
            else:
                chord_analysis.append(f"{note_name}")
        
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        print(f"  {start_time:5.1f}s - {end_time:5.1f}s: {' - '.join(chord_analysis)}")
    
    return {
        'tempo': tempo,
        'beat_times': beat_times,
        'key': estimated_key,
        'chroma_mean': chroma_mean,
        'duration': len(y)/sr
    }


def main():
    parser = argparse.ArgumentParser(description="Make audio faster without changing pitch using audiomentations")
    parser.add_argument("--speed", "-s", type=float, default=1.25, 
                       help="Speed multiplier (default: 1.25)")
    
    args = parser.parse_args()
    
    input_file = "Ride It - Jay Sean.mp3"
    output_file = "output.mp3"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    # Analyze original audio features
    print("="*60)
    original_features = analyze_audio_features(input_file, "Original Song")
    
    # Process the audio
    result = change_speed_preserve_pitch(input_file, args.speed)
    result.export(output_file, format="mp3")
    
    print(f"Done! Saved to {output_file}")
    
    # Analyze remixed audio features
    print("="*60)
    remix_features = analyze_audio_features(output_file, "Remixed Song")
    
    # Compare the results
    print("="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Original tempo: {original_features['tempo']:.2f} BPM")
    print(f"Remixed tempo:  {remix_features['tempo']:.2f} BPM")
    print(f"Tempo change:   {remix_features['tempo']/original_features['tempo']:.2f}x")
    print()
    print(f"Original duration: {original_features['duration']:.1f}s")
    print(f"Remixed duration:  {remix_features['duration']:.1f}s") 
    print(f"Duration ratio:    {remix_features['duration']/original_features['duration']:.2f}x")
    print()
    print(f"Original key: {original_features['key']}")
    print(f"Remixed key:  {remix_features['key']}")
    
    if original_features['key'] == remix_features['key']:
        print("✓ Key preserved successfully!")
    else:
        print("⚠ Key may have shifted slightly due to processing")
    
    print()
    print(f"Beat timing comparison:")
    print(f"Original beats: {len(original_features['beat_times'])}")
    print(f"Remixed beats:  {len(remix_features['beat_times'])}")
    
    # Show beat timing differences for first few beats
    min_beats = min(5, len(original_features['beat_times']), len(remix_features['beat_times']))
    print(f"\nFirst {min_beats} beat timing comparison:")
    for i in range(min_beats):
        orig_time = original_features['beat_times'][i]
        remix_time = remix_features['beat_times'][i]
        print(f"  Beat {i+1}: {orig_time:.3f}s → {remix_time:.3f}s (ratio: {remix_time/orig_time:.3f})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Lofi Remix Generator - Command Line Version
Usage: python lofi_nogradio.py <audio_file> [options]
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pydub import AudioSegment
import numpy as np
from audiomentations import TimeStretch
from audio_separator.separator import Separator
import shutil
import torch
from tqdm import tqdm

# Utility functions
def change_speed_preserve_pitch(audio_segment: AudioSegment, speed: float) -> AudioSegment:
    """Change speed while preserving pitch using time stretching"""
    mono = audio_segment.set_channels(1)
    samples = np.array(mono.get_array_of_samples(), dtype=np.float32)
    samples /= (2**15)
    # Time stretch
    ts = TimeStretch(min_rate=speed, max_rate=speed, p=1.0)
    stretched = ts(samples=samples, sample_rate=audio_segment.frame_rate)
    # Convert back
    stretched = (stretched * (2**15)).astype(np.int16)
    seg = AudioSegment(
        stretched.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        channels=1
    )
    if audio_segment.channels == 2:
        seg = AudioSegment.from_mono_audiosegments(seg, seg)
    return seg

def apply_delay(segment: AudioSegment, delay_ms: int, decay: float) -> AudioSegment:
    """Apply delay/echo effect"""
    echo = segment - (20 * decay)
    echo = AudioSegment.silent(duration=delay_ms) + echo
    return segment.overlay(echo)

def apply_reverb(segment: AudioSegment, room_sizes: list) -> AudioSegment:
    """Apply reverb effect using multiple echoes"""
    out = segment
    for delay_ms, decay in room_sizes:
        echo = segment - (30 * decay)
        echo = AudioSegment.silent(duration=delay_ms) + echo
        out = out.overlay(echo)
    return out

def adjust_volume(segment: AudioSegment, volume_db: float) -> AudioSegment:
    """Adjust volume of audio segment by specified dB"""
    return segment + volume_db

def check_cuda_availability():
    """Check if CUDA is available for GPU acceleration"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            return True, f"CUDA available: {device_name}"
        else:
            return False, "CUDA not available, using CPU"
    except ImportError:
        return False, "PyTorch not installed, using CPU"

def get_available_models():
    """Get list of available separation models"""
    try:
        result = subprocess.run(["audio-separator", "--list_models"], capture_output=True, text=True)
        if result.returncode == 0:
            model_lines = []
            for line in result.stdout.splitlines():
                if line.strip() and not line.startswith("Model") and not line.startswith("-"):
                    model_name = line.split()[0] if line.split() else line.strip()
                    if model_name:
                        model_lines.append(model_name)
            if model_lines:
                return model_lines
    except Exception as e:
        print(f"Error getting models: {e}")
    
    # Fallback to common models
    return ["htdemucs", "htdemucs_ft", "mdx_extra", "mdx23c", "spleeter"]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate lofi remixes from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lofi_nogradio.py song.mp3
  python lofi_nogradio.py song.wav --speed 0.7 --model htdemucs
  python lofi_nogradio.py song.mp3 --vocals-volume 2 --drums-volume -3
  python lofi_nogradio.py song.mp3 --enable-vocals-reverb --enable-drums-delay
        """
    )
    
    # Required arguments (but optional when listing models)
    parser.add_argument("audio_file", nargs='?', help="Input audio file path")
    
    # Model and speed
    parser.add_argument("--model", default="htdemucs", 
                       help="Separation model (default: htdemucs)")
    parser.add_argument("--speed", type=float, default=0.8, 
                       help="Speed multiplier (default: 0.8)")
    parser.add_argument("--output-dir", default="./lofi_stems",
                       help="Output directory for stems and final mix (default: ./lofi_stems)")
    
    # Volume controls
    parser.add_argument("--vocals-volume", type=float, default=0,
                       help="Vocals volume in dB (default: 0)")
    parser.add_argument("--drums-volume", type=float, default=-2,
                       help="Drums volume in dB (default: -2)")
    parser.add_argument("--bass-volume", type=float, default=2,
                       help="Bass volume in dB (default: 2)")
    parser.add_argument("--others-volume", type=float, default=-1,
                       help="Others volume in dB (default: -1)")
    
    # Delay effects
    parser.add_argument("--enable-vocals-delay", action="store_true",
                       help="Enable delay on vocals")
    parser.add_argument("--vocals-delay-ms", type=int, default=250,
                       help="Vocals delay in ms (default: 250)")
    parser.add_argument("--vocals-delay-decay", type=float, default=0.4,
                       help="Vocals delay decay (default: 0.4)")
    
    parser.add_argument("--enable-drums-delay", action="store_true",
                       help="Enable delay on drums")
    parser.add_argument("--drums-delay-ms", type=int, default=150,
                       help="Drums delay in ms (default: 150)")
    parser.add_argument("--drums-delay-decay", type=float, default=0.3,
                       help="Drums delay decay (default: 0.3)")
    
    parser.add_argument("--enable-bass-delay", action="store_true",
                       help="Enable delay on bass")
    parser.add_argument("--bass-delay-ms", type=int, default=200,
                       help="Bass delay in ms (default: 200)")
    parser.add_argument("--bass-delay-decay", type=float, default=0.5,
                       help="Bass delay decay (default: 0.5)")
    
    parser.add_argument("--enable-others-delay", action="store_true",
                       help="Enable delay on others")
    parser.add_argument("--others-delay-ms", type=int, default=300,
                       help="Others delay in ms (default: 300)")
    parser.add_argument("--others-delay-decay", type=float, default=0.3,
                       help="Others delay decay (default: 0.3)")
    
    # Reverb effects
    parser.add_argument("--enable-vocals-reverb", action="store_true", default=True,
                       help="Enable reverb on vocals (default: True)")
    parser.add_argument("--vocals-reverb", choices=["Small", "Medium", "Large"], default="Medium",
                       help="Vocals reverb preset (default: Medium)")
    
    parser.add_argument("--enable-drums-reverb", action="store_true",
                       help="Enable reverb on drums")
    parser.add_argument("--drums-reverb", choices=["Small", "Medium", "Large"], default="Small",
                       help="Drums reverb preset (default: Small)")
    
    parser.add_argument("--enable-bass-reverb", action="store_true",
                       help="Enable reverb on bass")
    parser.add_argument("--bass-reverb", choices=["Small", "Medium", "Large"], default="Medium",
                       help="Bass reverb preset (default: Medium)")
    
    parser.add_argument("--enable-others-reverb", action="store_true", default=True,
                       help="Enable reverb on others (default: True)")
    parser.add_argument("--others-reverb", choices=["Small", "Medium", "Large"], default="Large",
                       help="Others reverb preset (default: Large)")
    
    # Additional options
    parser.add_argument("--list-models", action="store_true",
                       help="List available separation models and exit")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # List models if requested
    if args.list_models:
        print("Available separation models:")
        models = get_available_models()
        for model in models:
            print(f"  - {model}")
        return
    
    # Validate input file (only if not listing models)
    if not args.audio_file:
        print("❌ Error: Audio file argument is required")
        sys.exit(1)
        
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file '{args.audio_file}' not found")
        sys.exit(1)
    
    # Print banner
    if not args.quiet:
        print("🎵 Lofi Remix Generator - Command Line Version")
        print("=" * 50)
        print(f"📁 Input file: {args.audio_file}")
        print(f"🚀 Model: {args.model}")
        print(f"⚡ Speed: {args.speed}x")
        print(f"📂 Output directory: {args.output_dir}")
        print()
    
    try:
        # Check CUDA availability
        cuda_available, cuda_status = check_cuda_availability()
        if not args.quiet:
            if cuda_available and not args.no_gpu:
                print(f"🚀 {cuda_status}")
            else:
                print(f"💻 {cuda_status}")
                if args.no_gpu:
                    print("   GPU acceleration disabled by --no-gpu flag")
            print()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        if not args.quiet:
            print(f"📁 Creating stems in: {args.output_dir}")
            print()
        
        # Initialize separator
        separator_kwargs = {
            "log_level": 40 if not args.quiet else 50,
            "output_dir": args.output_dir
        }
        
        # Note: CUDA support is handled automatically by the library
        # No need to explicitly set CUDA parameters
        
        separator = Separator(**separator_kwargs)
        
        if not args.quiet:
            print(f"📥 Loading model: {args.model}")
        separator.load_model(args.model)
        
        # Separate audio with progress bar
        if not args.quiet:
            print("\n🎵 Starting stem separation...")
            
        with tqdm(total=100, desc="Separating stems", unit="%", ncols=80, colour="green", disable=args.quiet) as pbar:
            pbar.update(30)
            output_files = separator.separate(args.audio_file)
            pbar.update(70)
            
        if not args.quiet:
            print(f"✅ Separation complete! Generated {len(output_files)} files")
            print()
        
        # Find separated files
        stems = {}
        for file_path in output_files:
            # Construct full path if only filename is returned
            if not os.path.isabs(file_path):
                full_path = os.path.join(args.output_dir, file_path)
            else:
                full_path = file_path
            
            # Check if file exists at the constructed path
            if not os.path.exists(full_path):
                # Try with just the output directory + basename
                full_path = os.path.join(args.output_dir, os.path.basename(file_path))
            
            filename = os.path.basename(full_path)
            
            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path) / (1024 * 1024)
                
                if "vocals" in filename.lower():
                    stems["vocals"] = full_path
                    if not args.quiet:
                        print(f"🎤 Vocals: {filename} ({file_size:.1f} MB)")
                elif "drums" in filename.lower():
                    stems["drums"] = full_path
                    if not args.quiet:
                        print(f"🥁 Drums: {filename} ({file_size:.1f} MB)")
                elif "bass" in filename.lower():
                    stems["bass"] = full_path
                    if not args.quiet:
                        print(f"🎸 Bass: {filename} ({file_size:.1f} MB)")
                elif "other" in filename.lower():
                    stems["others"] = full_path
                    if not args.quiet:
                        print(f"🎹 Others: {filename} ({file_size:.1f} MB)")
        
        if not stems:
            print("❌ Error: No stems were generated")
            sys.exit(1)
        
        # Define reverb presets
        reverb_presets = {
            "Small": [(50, 0.3), (100, 0.2)],
            "Medium": [(100, 0.4), (200, 0.3), (300, 0.2)],
            "Large": [(150, 0.5), (300, 0.4), (450, 0.3), (600, 0.2)]
        }
        
        # Process stems
        processed_stems = []
        stems_to_process = [k for k in ["vocals", "drums", "bass", "others"] if k in stems]
        
        if not args.quiet:
            print(f"\n🎛️ Processing {len(stems_to_process)} stems with effects...")
            print()
        
        with tqdm(total=len(stems_to_process), desc="Processing stems", unit="stem", 
                 ncols=80, colour="blue", disable=args.quiet) as effects_pbar:
            
            # Process vocals
            if "vocals" in stems:
                effects_pbar.set_description("🎤 Processing vocals")
                if not args.quiet:
                    print("🎤 Processing vocals...")
                
                seg = AudioSegment.from_file(stems["vocals"])
                effects = []
                
                seg = adjust_volume(seg, args.vocals_volume)
                effects.append(f"Volume: {args.vocals_volume:+.1f}dB")
                
                if args.enable_vocals_delay:
                    seg = apply_delay(seg, args.vocals_delay_ms, args.vocals_delay_decay)
                    effects.append(f"Delay: {args.vocals_delay_ms}ms")
                
                if args.enable_vocals_reverb:
                    seg = apply_reverb(seg, reverb_presets[args.vocals_reverb])
                    effects.append(f"Reverb: {args.vocals_reverb}")
                
                seg = change_speed_preserve_pitch(seg, args.speed)
                processed_stems.append(seg)
                
                if not args.quiet:
                    print(f"   ✅ Applied: {', '.join(effects)}")
                effects_pbar.update(1)
            
            # Process drums
            if "drums" in stems:
                effects_pbar.set_description("🥁 Processing drums")
                if not args.quiet:
                    print("🥁 Processing drums...")
                
                seg = AudioSegment.from_file(stems["drums"])
                effects = []
                
                seg = adjust_volume(seg, args.drums_volume)
                effects.append(f"Volume: {args.drums_volume:+.1f}dB")
                
                if args.enable_drums_delay:
                    seg = apply_delay(seg, args.drums_delay_ms, args.drums_delay_decay)
                    effects.append(f"Delay: {args.drums_delay_ms}ms")
                
                if args.enable_drums_reverb:
                    seg = apply_reverb(seg, reverb_presets[args.drums_reverb])
                    effects.append(f"Reverb: {args.drums_reverb}")
                
                seg = change_speed_preserve_pitch(seg, args.speed)
                processed_stems.append(seg)
                
                if not args.quiet:
                    print(f"   ✅ Applied: {', '.join(effects)}")
                effects_pbar.update(1)
            
            # Process bass
            if "bass" in stems:
                effects_pbar.set_description("🎸 Processing bass")
                if not args.quiet:
                    print("🎸 Processing bass...")
                
                seg = AudioSegment.from_file(stems["bass"])
                effects = []
                
                seg = adjust_volume(seg, args.bass_volume)
                effects.append(f"Volume: {args.bass_volume:+.1f}dB")
                
                if args.enable_bass_delay:
                    seg = apply_delay(seg, args.bass_delay_ms, args.bass_delay_decay)
                    effects.append(f"Delay: {args.bass_delay_ms}ms")
                
                if args.enable_bass_reverb:
                    seg = apply_reverb(seg, reverb_presets[args.bass_reverb])
                    effects.append(f"Reverb: {args.bass_reverb}")
                
                seg = change_speed_preserve_pitch(seg, args.speed)
                processed_stems.append(seg)
                
                if not args.quiet:
                    print(f"   ✅ Applied: {', '.join(effects)}")
                effects_pbar.update(1)
            
            # Process others
            if "others" in stems:
                effects_pbar.set_description("🎹 Processing others")
                if not args.quiet:
                    print("🎹 Processing others...")
                
                seg = AudioSegment.from_file(stems["others"])
                effects = []
                
                seg = adjust_volume(seg, args.others_volume)
                effects.append(f"Volume: {args.others_volume:+.1f}dB")
                
                if args.enable_others_delay:
                    seg = apply_delay(seg, args.others_delay_ms, args.others_delay_decay)
                    effects.append(f"Delay: {args.others_delay_ms}ms")
                
                if args.enable_others_reverb:
                    seg = apply_reverb(seg, reverb_presets[args.others_reverb])
                    effects.append(f"Reverb: {args.others_reverb}")
                
                seg = change_speed_preserve_pitch(seg, args.speed)
                processed_stems.append(seg)
                
                if not args.quiet:
                    print(f"   ✅ Applied: {', '.join(effects)}")
                effects_pbar.update(1)
        
        if not processed_stems:
            print("❌ Error: No stems were processed successfully")
            sys.exit(1)
        
        # Mix stems
        if not args.quiet:
            print("\n🎵 Mixing all processed stems...")
        
        with tqdm(total=len(processed_stems), desc="Mixing stems", unit="stem", 
                 ncols=80, colour="magenta", disable=args.quiet) as mix_pbar:
            final_mix = processed_stems[0]
            mix_pbar.update(1)
            
            for i, stem in enumerate(processed_stems[1:], 1):
                final_mix = final_mix.overlay(stem)
                mix_pbar.update(1)
        
        # Export final mix
        base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_lofi_remix_{args.model}.wav")
        
        if not args.quiet:
            print(f"\n💾 Exporting final remix to: {output_path}")
        
        with tqdm(total=1, desc="Exporting audio", unit="file", ncols=80, colour="cyan", disable=args.quiet) as export_pbar:
            final_mix.export(output_path, format="wav")
            export_pbar.update(1)
        
        # Final information
        final_size = os.path.getsize(output_path) / (1024 * 1024)
        
        if not args.quiet:
            print("✅ Export complete!")
            print(f"📊 Final remix size: {final_size:.1f} MB")
            print()
            print("🎉 Lofi remix generation complete!")
            print(f"🎵 Output file: {output_path}")
        else:
            print(output_path)  # Just print the output path in quiet mode
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error processing audio: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    
# # Simple usage with defaults
# python lofi_nogradio.py song.mp3

# # Custom speed and model
# python lofi_nogradio.py song.wav --speed 0.7 --model htdemucs

# # Adjust volumes for different stems
# python lofi_nogradio.py song.mp3 --vocals-volume 2 --drums-volume -3 --bass-volume 1

# # Enable effects
# python lofi_nogradio.py song.mp3 --enable-vocals-delay --enable-drums-reverb --enable-others-delay

# # Full customization
# python lofi_nogradio.py 'Kaho Na Kaho.mp3' --speed 0.8 --model htdemucs.yaml --vocals-volume 1 --enable-vocals-reverb --vocals-reverb Large --drums-volume -2   --bass-volume 3 --enable-bass-reverb --others-volume -1

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import csv
import torch
from PIL import Image
from diffusers import FluxPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, export_to_gif

# -----------------------------------------------------------------------------
# Configuration ─ adjust these if needed
# -----------------------------------------------------------------------------
CSV_PATH = "scene.csv"                    # Path to your CSV file
IMAGES_DIR = "images"                     # Directory where generated images will be saved
VIDEOS_DIR = "videos"                     # Directory where generated videos/GIFs will be saved

# Flux (image generation) settings
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FLUX_DTYPE = torch.bfloat16
FLUX_GUIDANCE_SCALE = 0.0
FLUX_HEIGHT = 576
FLUX_WIDTH = 1024
FLUX_STEPS = 4
FLUX_MAX_SEQ_LENGTH = 256

# Video (animation) settings
USE_GIF = False                            # If True, export as GIF; otherwise, export as MP4
MOTION_BUCKET_ID = 127                     # Motion control (1–255)
NOISE_AUG_STRENGTH = 0.1                   # Noise strength (0.0–1.0)
DECODE_CHUNK_SIZE = 3                      # Number of frames to decode at once (higher uses more VRAM)
NUM_FRAMES = 35                            # Total number of frames in the animation
FPS = 7                                    # Frames per second for output video/GIF
TARGET_RESOLUTION = (1024, 576)            # Desired (width, height) for the video pipeline
VERSION = "svd"                            # "svd" for 14 fps–trained model, "svdxt" for 25 fps–trained

# Seed settings (shared between image and video generation for consistency)
BASE_SEED = 42

# -----------------------------------------------------------------------------
# Utility: Resize + center‐crop to TARGET_RESOLUTION, preserving aspect ratio
# -----------------------------------------------------------------------------
def resize_image(image: Image.Image, output_size=(1024, 576)) -> Image.Image:
    target_w, target_h = output_size
    img_w, img_h = image.size
    target_aspect = target_w / target_h
    img_aspect = img_w / img_h

    # If already correct size, return as-is
    if img_w == target_w and img_h == target_h:
        return image

    # Resize while preserving aspect ratio
    if img_aspect > target_aspect:
        # Wider than target: match height, then crop width
        new_h = target_h
        new_w = int(new_h * img_aspect)
        resized = image.resize((new_w, new_h), resample=Image.LANCZOS)
        left = (new_w - target_w) // 2
        top = 0
    else:
        # Taller than target: match width, then crop height
        new_w = target_w
        new_h = int(new_w / img_aspect)
        resized = image.resize((new_w, new_h), resample=Image.LANCZOS)
        left = 0
        top = (new_h - target_h) // 2

    right = left + target_w
    bottom = top + target_h
    return resized.crop((left, top, right, bottom))


def main():
    # -----------------------------------------------------------------------------
    # 1. Ensure output directories exist
    # -----------------------------------------------------------------------------
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------------
    # 2. Read CSV and generate images via FluxPipeline
    # -----------------------------------------------------------------------------
    print("Loading FluxPipeline for image generation...")
    flux_pipe = FluxPipeline.from_pretrained(FLUX_MODEL_ID, torch_dtype=FLUX_DTYPE)
    flux_pipe.enable_model_cpu_offload()

    data = csv.reader(open(CSV_PATH, "r"))
    next(data)  # Skip header row
    for scene_number, row in enumerate(data, start=1):
        prompt = row[1]
        print(f"[Flux] Generating image for scene {scene_number}: \"{prompt}\"")

        # Use a reproducible seed per scene
        flux_generator = torch.Generator(device="cpu").manual_seed(BASE_SEED + scene_number)

        output = flux_pipe(
            prompt=prompt,
            guidance_scale=FLUX_GUIDANCE_SCALE,
            height=FLUX_HEIGHT,
            width=FLUX_WIDTH,
            num_inference_steps=FLUX_STEPS,
            max_sequence_length=FLUX_MAX_SEQ_LENGTH,
            generator=flux_generator
        )
        image = output.images[0]
        image_path = os.path.join(IMAGES_DIR, f"scene_{scene_number}.png")
        image.save(image_path)
        print(f"[Flux] Saved image to: {image_path}")

    # Unload FluxPipeline and clear any GPU memory (if used)
    print("Unloading FluxPipeline and clearing GPU cache...")
    del flux_pipe
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------------
    # 3. Verify CUDA availability for video generation
    # -----------------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA (GPU) is required to run Stable Video Diffusion pipelines.")

    device = "cuda"

    # -----------------------------------------------------------------------------
    # 4. Load StableVideoDiffusionPipeline(s) into VRAM
    # -----------------------------------------------------------------------------
    print("Loading Stable Video Diffusion pipelines into VRAM (fp16)...")
    fps25_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "vdo/stable-video-diffusion-img2vid-xt-1-1",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    fps14_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    print("Pipelines loaded.")

    # Choose which pipeline to use
    if VERSION.lower() == "svdxt":
        chosen_pipe = fps25_pipe
        chosen_fps = 25
        print("[Video] Using SVD-XT (25 fps–trained) pipeline.")
    else:
        chosen_pipe = fps14_pipe
        chosen_fps = 14
        print("[Video] Using SVD (14 fps–trained) pipeline.")

    # -----------------------------------------------------------------------------
    # 5. For each generated image, produce a video (or GIF)
    # -----------------------------------------------------------------------------
    # List all images by scene number (assuming no gaps)
    image_files = sorted(
        f for f in os.listdir(IMAGES_DIR) if f.startswith("scene_") and f.endswith(".png")
    )

    for scene_filename in image_files:
        # Extract scene number from filename
        basename = os.path.splitext(scene_filename)[0]  # "scene_1"
        scene_number = int(basename.split("_")[1])
        image_path = os.path.join(IMAGES_DIR, scene_filename)

        print(f"[Video] Processing scene {scene_number} image at: {image_path}")
        init_image = Image.open(image_path)
        if init_image.mode == "RGBA":
            init_image = init_image.convert("RGB")
        init_image = resize_image(init_image, output_size=TARGET_RESOLUTION)

        # Set up reproducible seed per scene
        torch.manual_seed(BASE_SEED + scene_number)
        video_generator = torch.Generator(device=device).manual_seed(BASE_SEED + scene_number)

        print(f"[Video] Generating {NUM_FRAMES} frames (this can take a couple of minutes)...")
        output = chosen_pipe(
            init_image,
            decode_chunk_size=DECODE_CHUNK_SIZE,
            generator=video_generator,
            motion_bucket_id=MOTION_BUCKET_ID,
            noise_aug_strength=NOISE_AUG_STRENGTH,
            num_frames=NUM_FRAMES,
        )
        video_frames = output.frames[0]
        print(f"[Video] Generated {len(video_frames)} frames (resolution: {TARGET_RESOLUTION[0]}×{TARGET_RESOLUTION[1]}).")

        # Save as MP4 or GIF
        if USE_GIF:
            gif_path = os.path.join(VIDEOS_DIR, f"{basename}.gif")
            print(f"[Video] Exporting to GIF at '{gif_path}' (fps={chosen_fps})...")
            export_to_gif(image=video_frames, output_gif_path=gif_path, fps=chosen_fps)
            print(f"[Video] GIF saved: {gif_path}")
        else:
            video_path = os.path.join(VIDEOS_DIR, f"{basename}.mp4")
            print(f"[Video] Exporting to MP4 at '{video_path}' (fps={chosen_fps})...")
            export_to_video(video_frames, video_path, fps=chosen_fps)
            print(f"[Video] MP4 saved: {video_path}")

    print("All scenes processed. Animation pipeline complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import cv2
from tqdm import tqdm

def create_video(image_dir, output_file, fps=10, extension="png", resize=None, sort_reverse=False):
    """
    Creates an MP4 video from image files in a directory.
    
    Parameters:
        image_dir (str): Directory containing the images.
        output_file (str): Path to the output MP4 file.
        fps (int): Frames per second.
        extension (str): File extension of the images (e.g., 'png', 'jpg').
        resize (tuple): Optional tuple (width, height) to resize images.
        sort_reverse (bool): Reverse the sorting order of the images.
    """
    # 1. Resolve path and search for images
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.", file=sys.stderr)
        return False
        
    search_path = os.path.join(image_dir, f"*.{extension}")
    image_paths = sorted(glob.glob(search_path), reverse=sort_reverse)
    
    if not image_paths:
        print(f"Error: No files matching '*.{extension}' found in '{image_dir}'.", file=sys.stderr)
        return False
        
    print(f"Found {len(image_paths)} images in '{image_dir}'.")
    
    # 2. Read first image to determine frame size
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"Error: Could not read the first image '{image_paths[0]}'.", file=sys.stderr)
        return False
        
    h, w, c = first_image.shape
    if resize:
        frame_width, frame_height = resize
        print(f"Images will be resized from {w}x{h} to {frame_width}x{frame_height}.")
    else:
        frame_width, frame_height = w, h
        print(f"Video resolution: {frame_width}x{frame_height} (auto-detected from first image).")
        
    # 3. Initialize VideoWriter
    # 'mp4v' is widely supported for MP4 container.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    if not video_writer.isOpened():
        print("Error: Could not open VideoWriter. Check output path and permissions.", file=sys.stderr)
        return False
        
    try:
        print(f"Writing video to '{output_file}' at {fps} FPS...")
        for path in tqdm(image_paths, desc="Processing frames"):
            frame = cv2.imread(path)
            if frame is None:
                print(f"\nWarning: Skipping unreadable image '{path}'", file=sys.stderr)
                continue
                
            # Resize if dimensions differ or if custom resize is specified
            fh, fw, _ = frame.shape
            if resize or (fw != frame_width or fh != frame_height):
                frame = cv2.resize(frame, (frame_width, frame_height))
                
            video_writer.write(frame)
            
        print("\nVideo writing completed successfully.")
        return True
        
    except Exception as e:
        print(f"\nAn error occurred during video creation: {e}", file=sys.stderr)
        return False
    finally:
        video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an MP4 video from a directory of images.")
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default="./aia193_plot",
        help="Path to the directory containing images (default: ./aia193_plot)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./aia193_video.mp4",
        help="Output video file path (default: ./aia193_video.mp4)"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=120, 
        help="Frames per second for the output video (default: 10)"
    )
    parser.add_argument(
        "--ext", 
        type=str, 
        default="png", 
        help="Image file extension to search for (default: png)"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=None, 
        help="Optional: Target width of the video. If set, --height must also be set."
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=None, 
        help="Optional: Target height of the video. If set, --width must also be set."
    )
    parser.add_argument(
        "--reverse", 
        action="store_true", 
        help="Reverse chronological/alphabetical order of images"
    )

    args = parser.parse_args()
    
    # Validate resolution arguments
    resize_tuple = None
    if (args.width is not None) or (args.height is not None):
        if args.width is None or args.height is None:
            print("Error: Both --width and --height must be specified if you want to resize.", file=sys.stderr)
            sys.exit(1)
        resize_tuple = (args.width, args.height)
        
    success = create_video(
        image_dir=args.image_dir,
        output_file=args.output,
        fps=args.fps,
        extension=args.ext,
        resize=resize_tuple,
        sort_reverse=args.reverse
    )
    
    if not success:
        sys.exit(1)

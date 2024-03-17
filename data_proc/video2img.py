# decode_video_to_images.py
import cv2
import os

def decode_video(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save each frame to the output folder
        frame_filename = os.path.join(output_folder, f"frame{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames and saved to {output_folder}")

# Example usage
if __name__ == "__main__":
    video_path = "/home/jc/xinrun/TestData/2024-03-15-16-14-05/overview.mp4"
    output_folder = "/home/jc/xinrun/TestData/2024-03-15-16-14-05/raw"
    decode_video(video_path, output_folder)

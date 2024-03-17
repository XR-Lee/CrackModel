# regenerate_video_from_images.py
import cv2
import os

def images_to_video(input_folder, output_video_path, fps=30):
    images = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images.sort()
    
    # Assuming all images are the same size, get the dimensions of the first image
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for image in images:
        frame = cv2.imread(os.path.join(input_folder, image))
        out.write(frame)  # Write out frame to video
    
    out.release()
    print(f"Video created at {output_video_path} with {len(images)} frames.")

# Example usage
if __name__ == "__main__":
    input_folder = "/home/jc/xinrun/TestData/2024-03-15-16-14-05/vis"
    output_video_path = "/home/jc/xinrun/TestData/2024-03-15-16-14-05/vis_video.mp4"
    images_to_video(input_folder, output_video_path)

from PIL import Image
import os
from tqdm import tqdm
# Function to resize image while maintaining aspect ratio
def resize_image(input_image_path, output_image_path, base_width=None, base_height=None):
    with Image.open(input_image_path) as img:
        # Calculate new dimensions
        w_percent = (base_width / float(img.size[0])) if base_width else (base_height / float(img.size[1]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        w_size = int((float(img.size[0]) * float(w_percent)))

        # Resize image
        img = img.resize((base_width or w_size, base_height or h_size))
        img.save(output_image_path,format='png')

if __name__ == "__main__":
    # Define the source and destination folders
    source_folder = '/home/iix5sgh/workspace/crack/dataset/crack_dataset/image_wh1600'
    destination_folder = '/home/iix5sgh/workspace/crack/dataset/crack_dataset/image_wh400'

    # Set either the desired width or height, and set the other as None
    base_width = None  # Set your desired width, or set to None
    base_height = 400  # Set your desired height, or set to None

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Resize each image in the source folder and save to the destination folder
    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add/check other file types if needed
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename.split('.')[0] + '.png')
            resize_image(source_path, destination_path, base_width, base_height)

import os
import cv2
from tqdm import tqdm 

def crop_images(folder_path,save_path, patch_size=400, overlap_size=0):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') ]

    for image_file in tqdm(image_files):
        # Read the image
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        # Get the image dimensions
        height, width, _ = image.shape
        
        # Calculate the number of patches in each dimension
 
        num_patches_x = width // (patch_size-overlap_size )
        num_patches_y = height //( patch_size-overlap_size)
        
        # print(width)
        # print(patch_size-overlap_size)
        # print(num_patches_x)
        
        # print(height)
        # print(num_patches_y)


        # Crop the image into patches
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                # Calculate the patch coordinates
                x = j * (patch_size - overlap_size)
                y = i * (patch_size - overlap_size)
                x_end = x + patch_size 
                y_end = y + patch_size 
                if x_end > width:
                    break
                if y_end >height:
                    break
                # Crop the patch
                patch = image[max(0, y):min(y_end, height), max(0, x):min(x_end, width)]
                
                
                name, extension = os.path.splitext(image_file)
                # Save the patch
                patch_filename = f"{name}_{i}_{j}.jpeg"
                patch_path = os.path.join(save_path, patch_filename)
                cv2.imwrite(patch_path, patch)
                
def main_function():
        # Usage example
    folder_path = '/home/iix5sgh/workspace/lerf/data/crack_scene_1/images/'
    save_folder ='/home/iix5sgh/workspace/lerf/data/crack_scene_1/crops/'
    crop_images(folder_path,save_folder, patch_size=400, overlap_size=160)



if __name__ == "__main__":
    main_function()

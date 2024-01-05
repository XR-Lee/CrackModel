import os
import numpy as np
from PIL import Image, ImageOps, ImageChops
from tqdm import tqdm


def colorize_mask(mask, color):
    """ Colorize the mask """
    colored_mask = ImageOps.colorize(mask, black=(0, 0, 0), white=color)
    return colored_mask.convert('RGBA')

def create_mask(image, target_value=200):
    """ Create a mask for pixels equal to the target_value """
    mask = Image.new(mode="RGBA", size=image.size, color=(0,0,0,0))
    for x in range(image.width):
        for y in range(image.height):
            # print(image.getpixel((x, y)))
            # if image.getpixel((x, y))[0] >= target_value:
                # mask.putpixel((x, y), (255,0,0,255))
            mask.putpixel((x, y),image.getpixel((x, y)) )
                # print(mask.getpixel((x, y)))
    return mask


def overlay_masks(image_folder, mask_folder, output_folder):
    """
    Overlay mask images onto corresponding raw images.

    :param image_folder: Path to the folder containing raw images.
    :param mask_folder: Path to the folder containing mask images.
    :param output_folder: Path to the folder where output images will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all images in the image folder
    for image_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)
        
        name, extension = os.path.splitext(image_name)
        
        png_mask_name = name + ".png"
        print(png_mask_name)
        # Check if the corresponding mask exists
        # jpg_mask_path = os.path.join(mask_folder, image_name)
        png_mask_path = os.path.join(mask_folder,png_mask_name)  
        mask_path = png_mask_path
        
        if os.path.exists(mask_path):
            # Open the image and the mask
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path).convert("L")  # Ensure mask is in RGBA
            combined_image_path = os.path.join(output_folder, png_mask_name)
             
            alpha = mask
            colored_mask = ImageOps.colorize(mask, black='black', white=(255,0,0)).convert('RGBA')
            colored_mask.putalpha(alpha)
            # print(list(colored_mask.getdata()))
            # Overlay the mask onto the image
            combined_image = Image.alpha_composite( image.convert("RGBA"), colored_mask)
            
           
            combined_image.save(combined_image_path)



def main_function():
    # Note: The paths need to be adjusted to the actual folders on your system.
    image_folder_path = '/home/iix5sgh/workspace/lerf/data/crack_scene_1/crops/'
    mask_folder_path = '/home/iix5sgh/workspace/lerf/data/crack_scene_1/infer/'
    output_folder_path = '/home/iix5sgh/workspace/crack/mask_visual_crops/'
    
    
    # Call the function
    overlay_masks(image_folder_path, mask_folder_path, output_folder_path)


if __name__ == "__main__":
    main_function()

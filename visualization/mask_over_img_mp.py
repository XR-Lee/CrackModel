import os
import multiprocessing
from PIL import Image, ImageOps
from tqdm import tqdm

def process_image(image_path, mask_path, output_path):
    """Process a single image."""
    if os.path.exists(mask_path):
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert("L")  # Ensure mask is in L mode
        colored_mask = ImageOps.colorize(mask, black='black', white=(255,0,0)).convert('RGBA')
        colored_mask.putalpha(mask)  # Use the original mask as alpha channel
        combined_image = Image.alpha_composite(image.convert("RGBA"), colored_mask)
        combined_image.save(output_path)

def overlay_masks(image_folder, mask_folder, output_folder):
    """Overlay masks on images using multiprocessing."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a list of tasks
    tasks = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        name, _ = os.path.splitext(image_name)
        png_mask_name = name + ".png"
        png_mask_path = os.path.join(mask_folder, png_mask_name)
        output_path = os.path.join(output_folder, png_mask_name)
        
        # Add the task
        tasks.append((image_path, png_mask_path, output_path))

    # Use Pool to process images in parallel
    pool = multiprocessing.Pool()
    for task in tqdm(tasks):
        pool.apply_async(process_image, task)
    
    pool.close()
    pool.join()

def main_function():
    image_folder_path = '/home/jc/xinrun/TestData/2024-03-15-16-14-05/raw'
    mask_folder_path = '/home/jc/xinrun/TestData/2024-03-15-16-14-05/mask_v2'
    output_folder_path = '/home/jc/xinrun/TestData/2024-03-15-16-14-05/vis_v2'
    overlay_masks(image_folder_path, mask_folder_path, output_folder_path)

if __name__ == "__main__":
    main_function()

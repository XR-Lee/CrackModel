from load_SAM import *
from load_Images import ImageLoader
from mask_process import *
import matplotlib.pyplot as plt
import os
import tqdm

SAMPredictor = load_SAM()

root_path= "/home/jc/jiapeng/bridge_circle2_20240319/2757/"
image_sub_folder = "images2/"
mask_sub_folder = "mask/"
refine_sub_folder = "refine/"

files = os.listdir(root_path+image_sub_folder)

# Filter and modify the file names
files = [os.path.splitext(file)[0] if file.endswith(('.png', '.jpg')) else file for file in files]

for file_name in tqdm.tqdm(files):
    image_path = root_path + image_sub_folder + file_name + ".jpg"
    mask_path = root_path + mask_sub_folder + file_name + ".png"
    image_loader = ImageLoader(image_path, mask_path)

    clusters = image_loader.cluster_EDT()
    crop_bundles = image_loader.get_crop_bundle()

    if len(crop_bundles)==0:
        continue
    
    n = 0 # number of crops 
    for crop_bundle in crop_bundles:
        n += 1
        pts, label = EDT_to_pts(crop_bundle[2])

        if len(pts) == 0:
            continue

        SAMPredictor.set_image(crop_bundle[0])


        masks, scores, logits = SAMPredictor.predict(
            point_coords=pts,
            point_labels=label,
            multimask_output=False,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            fig,axe = plt.subplots(1,2, figsize=(8,8))
            axe[0].imshow(crop_bundle[2])
            axe[1].imshow(crop_bundle[0])
            show_mask(mask,axe[1], plt.gca())
            show_points(pts, label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=10)
            plt.axis('off')
        
            plt.savefig(root_path+refine_sub_folder+file_name+"_"+str(n)+".png")
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

    # Prompt Rejector : no proper prompts
    if len(crop_bundles)==0:
        continue
    
    n = 0 # number of crops 
    for crop_bundle in crop_bundles:
        n += 1
        pts, label = EDT_to_pts(crop_bundle[2])

        # Point Rejector : too few pts will have instable performance 
        if len(pts) <= 3:
            continue

        SAMPredictor.set_image(crop_bundle[0])

        masks, scores, logits = SAMPredictor.predict(
            point_coords=pts,
            point_labels=label,
            multimask_output=True,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):

            reject_flag = False

            fig,axe = plt.subplots(1,3, figsize=(8,6))
            image1 = axe[0].imshow(crop_bundle[2],cmap='viridis')
            image2 = axe[2].imshow(crop_bundle[0])


            color = np.array([255/255])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) 
            EDT = mask_EDT(mask_image)
            image3 = axe[1].imshow(EDT,cmap='viridis')
            # print(get_EDT_max(EDT))

            if get_EDT_max(EDT) > 3*get_EDT_max(crop_bundle[2]):
                reject_flag = True

            show_mask(mask,axe[2])
            show_points(pts, label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=10)
            fig.colorbar(image1, ax=axe[0],fraction=0.046, pad=0.04)
            fig.colorbar(image3, ax=axe[1],fraction=0.046, pad=0.04)
            # fig.colorbar(image3, ax=axe[2],fraction=0.046, pad=0.14)
            axe[1].axis('off')
            axe[2].axis('off')

            if reject_flag:
                plt.savefig(root_path+refine_sub_folder+"reject_"+file_name+"_"+str(n)+"_"+str(i)+".png")
            else:
                plt.savefig(root_path+refine_sub_folder+file_name+"_"+str(n)+"_"+str(i)+".png")
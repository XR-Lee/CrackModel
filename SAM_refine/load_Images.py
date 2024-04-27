
import cv2
from mask_process import mask_EDT

class ImageLoader:

    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        self.load_raw()

    def load_raw(self):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path,cv2.IMREAD_GRAYSCALE)
        EDT = mask_EDT(mask)
        # print(mask.shape)
        # print(image.shape)
        # print(EDT.shape)
        self.image = image
        self.mask = mask
        self.EDT = EDT
        return  
    
    def cluster_EDT(self):
        # Threshold the mask to identify pixels with values within the specified range
        binary_mask = cv2.inRange(self.mask, 1, 255)
        # Find connected components in the binary mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        # Create a dictionary to store the size of each cluster
        cluster_bounding_boxes = []

        # Iterate through each component
        for label in range(1, num_labels):  # Start from 1 to ignore background label
            # Get the size of the component (area)
            x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], \
                 stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
            # Store the bounding box coordinates in the dictionary
            if w >100 and h > 100:
                cluster_bounding_boxes.append([x, y, w, h])

        ## Print the bounding box coordinates for each cluster
        # for bbox in cluster_bounding_boxes:
            # print(f"Bounding Box {bbox}")

        self.clusters = cluster_bounding_boxes
        return self.clusters

    def get_crop_bundle(self):
        if self.clusters is not None and len(self.clusters) > 0:
            crop_bundles = []
            for cluster in self.clusters:
                x, y, w, h = cluster
                # print(cluster)
                # Crop the image
                if x-200 < 0:
                    x = 200
                if y-200 < 0:
                    y = 200
                crop_frame = [y-200,y+h+200,x-200,x+w+200]
                image = self.image[crop_frame[0]:crop_frame[1], crop_frame[2]:crop_frame[3]]
                mask = self.mask[crop_frame[0]:crop_frame[1], crop_frame[2]:crop_frame[3]]
                EDT = self.EDT[crop_frame[0]:crop_frame[1], crop_frame[2]:crop_frame[3]]
                crop_bundles.append([image, mask, EDT])
            return crop_bundles
        else:
            return []

    def get_image(self):
        return self.image
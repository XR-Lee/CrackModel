import os
from PIL import Image
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from torchvision import transforms
from dataloaders import custom_transforms as tr
import cv2
import numpy as np

model = DeepLab(num_classes=2,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn= None,
                        freeze_bn=False)

model = torch.nn.DataParallel(model, device_ids=[0])
patch_replication_callback(model)
model = model.cuda()

weight_name="/home/jc/xinrun/CrackModel/DeepLab/run/crack/deeplab-resnet/experiment_10/checkpoint140.pth.tar"

checkpoint = torch.load(weight_name)
model.module.load_state_dict(checkpoint['state_dict'])

model.eval()
print('Model loaded')
composed_transforms = transforms.Compose([
            tr.Normalize(),
            tr.Ignore_label(),
            tr.ToTensor()])



directory = '/home/jc/xinrun/TestData/2024-03-15-16-14-05/raw/'
output_dir = '/home/jc/xinrun/TestData/2024-03-15-16-14-05/mask_v2/'
# file = open('/home/iix5sgh/workspace/crack/dataset/Final_Dataset/test.txt', 'r')
files_in_directory = os.listdir(directory)
for rgb_img_path in files_in_directory:
    # delete last four characters
    rgb_img_path = rgb_img_path[:-4]
    # img = Image.open('/home/iix5sgh/workspace/crack/dataset/Final_Dataset/Images/' + rgb_img_path + '.png').convert('RGB')
    img = Image.open(directory + rgb_img_path + '.jpg').convert('RGB')
    sample = {'image': img, 'label': img}
    sample = composed_transforms(sample)
    img = sample['image']
    img = img.cuda()
    # img = torch.unsqueeze(img, 0)
    img = img.repeat(1, 1, 1, 1)
    with torch.no_grad():
        output = model(img)

    pred = torch.max(output[:3], 1)[1].detach().cpu().numpy()
    # print('shape of output: ', pred[0].shape)


    # print('pred[0] max value is: ', np.max(pred[0]))
    cv2.imwrite(output_dir  + rgb_img_path + '.png', pred[0] * 255)
    # im.save("paper_images/CFD_001.png")

    print('Image saved ', rgb_img_path)
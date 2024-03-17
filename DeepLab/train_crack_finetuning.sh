CUDA_VISIBLE_DEVICES=0 \
python3 train_feat.py \
--backbone resnet \
--lr 0.02 \
--workers 4 \
--epochs 300 \
--batch-size 2 \
--gpu-ids 0 \
--dataset crack \
--start_epoch 0 \
--eval-interval 5 \
--base-size 1024 \
--crop-size 448 \
--resume /home/jc/xinrun/CrackModel/DeepLab/run/crack/deeplab-resnet/experiment_6/checkpoint31.pth.tar \
# --resume /home/jc/Desktop/traintest/CrackModel/DeepLab/run/crack/deeplab-resnet/experiment_10/checkpoint200.pth.tar \
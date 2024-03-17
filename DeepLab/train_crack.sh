CUDA_VISIBLE_DEVICES=0 \
python3 train_feat.py \
--backbone resnet \
--lr 0.02 \
--workers 4 \
--epochs 50 \
--batch-size 32 \
--gpu-ids 0 \
--dataset crack \
--start_epoch 0 \
--eval-interval 1 \
--base-size 448 \
--crop-size 448 \
# --resume /home/jc/xinrun/CrackModel/DeepLab/model_weights/20240311_checkpoint3.pth.tar \
# --resume /home/jc/Desktop/traintest/CrackModel/DeepLab/run/crack/deeplab-resnet/experiment_10/checkpoint200.pth.tar \
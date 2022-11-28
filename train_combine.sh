target_dataset=voc2007
cfg_path=configs/baselines/vgg16_voc2007_combine.yaml
CUDA_VISIBLE_DEVICES=0 /home/zenghao/anaconda3/envs/pytorch/bin/python tools/train_net_step.py --dataset $target_dataset \
    --cfg $cfg_path --bs 1 --nw 4 --iter_size 1
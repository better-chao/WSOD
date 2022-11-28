target_dataset=voc2007
cfg_path=configs/baselines/vgg16_voc2007_adl.yaml
CUDA_VISIBLE_DEVICES=0 python tools/train_ADL_step.py --dataset $target_dataset \
    --cfg $cfg_path --bs 1 --nw 10 --iter_size 10
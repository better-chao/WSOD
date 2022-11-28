
model_path=Outputs/vgg16_voc2007_combine/Dec22-06-15-08_compute08_step/ckpt/model_step149999.pth
output_dir=Outputs/vgg16_voc2007_combine/Dec22-06-15-08_compute08_step/test/model_step149999
cfg_path=configs/baselines/vgg16_voc2007_combine.yaml
target_dataset=voc2007test

CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --cfg $cfg_path \
    --load_ckpt $model_path \
    --dataset $target_dataset --vis

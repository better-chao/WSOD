
model_path=Outputs/vgg16_voc2012/Apr30-18-41-50_compute15_step/ckpt/model_step37499.pth
output_dir=Outputs/vgg16_voc2012/Apr30-18-41-50_compute15_step/test/model_step37499
cfg_path=configs/baselines/vgg16_voc2012.yaml
target_dataset=voc2012test

CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --cfg $cfg_path \
    --load_ckpt $model_path \
    --dataset $target_dataset --vis

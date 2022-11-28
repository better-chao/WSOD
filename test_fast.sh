
model_path=Outputs/vgg16_voc2007_fast/Apr28-15-45-43_compute05_step/ckpt/model_step18749.pth
output_dir=Outputs/vgg16_voc2007_fast/Apr28-15-45-43_compute05_step/test/model_step18749
cfg_path=configs/baselines/vgg16_voc2007_fast.yaml
target_dataset=voc2007trainval

CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --cfg $cfg_path \
    --load_ckpt $model_path \
    --dataset $target_dataset --vis

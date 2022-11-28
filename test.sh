
model_path=Outputs/vgg16_voc2007_more/Apr06-08-15-09_compute05_step/ckpt/model_step139999.pth
output_dir=Outputs/vgg16_voc2007_more/Apr06-08-15-09_compute05_step/test/model_step139999
cfg_path=configs/baselines/vgg16_voc2007_more.yaml
target_dataset=voc2007trainval

CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --cfg $cfg_path \
    --load_ckpt $model_path \
    --dataset $target_dataset --vis

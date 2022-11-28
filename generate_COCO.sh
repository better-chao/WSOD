
model_path=Outputs/vgg16_voc2007_more/Feb19-13-36-25_compute01_step/ckpt/model_step149999.pth
cfg_path=configs/baselines/vgg16_voc2007_more.yaml
target_dataset=voc_2007_trainval
proposal_file=data/selective_search_data/voc_2007_trainval.pkl

CUDA_VISIBLE_DEVICES=0 python tools/generator_COCO_label.py --cfg $cfg_path \
    --load_ckpt $model_path \
    --dataset $target_dataset \
    --proposal_file $proposal_file

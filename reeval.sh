output_dir=Outputs/vgg16_voc2007_fast/Apr28-15-45-43_compute05_step/test/model_step18749
target_dataset=voc2007test
cfg_path=configs/baselines/vgg16_voc2007_fast.yaml
CUDA_VISIBLE_DEVICES=0 python tools/reeval.py --result_path $output_dir/detections.pkl \
    --dataset $target_dataset --cfg $cfg_path > $output_dir/test.txt
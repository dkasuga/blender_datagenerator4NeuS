#!/bin/zsh
BLEND_FILE=$1
BLEND_SCRIPT=$2
OUTPUT_DIR=$3
VIEWS_NUM=$4
WIDTH=$5
HEIGHT=$6
RADIUS=$7

COARSE_MASK_DIR="${OUTPUT_DIR}/coarse_mask"
MASK_DIR="${OUTPUT_DIR}/mask"

./blender2.91.2/blender -P setgpu.py -b $BLEND_FILE -P $BLEND_SCRIPT -- --output_dir $OUTPUT_DIR --views $VIEWS_NUM --width $WIDTH --height $HEIGHT --radius $RADIUS --random_views --upper_views --png
python refine_maskimage.py $COARSE_MASK_DIR $MASK_DIR
python preprocess/preprocess_cameras.py --source_dir $OUTPUT_DIR

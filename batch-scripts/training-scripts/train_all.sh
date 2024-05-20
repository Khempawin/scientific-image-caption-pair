#!/bin/bash
BASE_MODEL_DIR="/home/horton/datasets/meta-scir/models"
BASE_DATA_DIR="/home/horton/datasets/meta-scir"


# # miread
# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/miread-finetune" \
#     --model_name_or_path "$BASE_MODEL_DIR/miread" \
#     --data_dir "$BASE_DATA_DIR/dataset-plain" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/miread-finetune/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/miread-section" \
#     --model_name_or_path "$BASE_MODEL_DIR/miread" \
#     --data_dir "$BASE_DATA_DIR/dataset-section" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/miread-section/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/miread-meta" \
#     --model_name_or_path "$BASE_MODEL_DIR/miread" \
#     --data_dir "$BASE_DATA_DIR/dataset-title-section" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/miread-meta/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/miread-special-token-meta" \
#     --model_name_or_path "$BASE_MODEL_DIR/miread-special-token-base" \
#     --data_dir "$BASE_DATA_DIR/dataset-meta-special-token" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/miread-special-token-meta/checkpoint-*"

# # roberta
# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/roberta-finetune" \
#     --model_name_or_path "$BASE_MODEL_DIR/roberta" \
#     --data_dir "$BASE_DATA_DIR/dataset-plain" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/roberta-finetune/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/roberta-section" \
#     --model_name_or_path "$BASE_MODEL_DIR/roberta" \
#     --data_dir "$BASE_DATA_DIR/dataset-section" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/roberta-section/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/roberta-meta" \
#     --model_name_or_path "$BASE_MODEL_DIR/roberta" \
#     --data_dir "$BASE_DATA_DIR/dataset-title-section" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/roberta-meta/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/roberta-special-token-meta" \
#     --model_name_or_path "$BASE_MODEL_DIR/roberta-special-token-base" \
#     --data_dir "$BASE_DATA_DIR/dataset-meta-special-token" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/roberta-special-token-meta/checkpoint-*"

# # scibert
# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/scibert-finetune" \
#     --model_name_or_path "$BASE_MODEL_DIR/scibert" \
#     --data_dir "$BASE_DATA_DIR/dataset-plain" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/scibert-finetune/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/scibert-section" \
#     --model_name_or_path "$BASE_MODEL_DIR/scibert" \
#     --data_dir "$BASE_DATA_DIR/dataset-section" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/scibert-section/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/scibert-meta" \
#     --model_name_or_path "$BASE_MODEL_DIR/scibert" \
#     --data_dir "$BASE_DATA_DIR/dataset-title-section" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/scibert-meta/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/scibert-special-token-meta" \
#     --model_name_or_path "$BASE_MODEL_DIR/scibert-special-token-base" \
#     --data_dir "$BASE_DATA_DIR/dataset-meta-special-token" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/scibert-special-token-meta/checkpoint-*"

# specter2
# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/specter2-finetune" \
#     --model_name_or_path "$BASE_MODEL_DIR/specter2" \
#     --data_dir "$BASE_DATA_DIR/dataset-plain" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/specter2-finetune/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/specter2-section" \
#     --model_name_or_path "$BASE_MODEL_DIR/specter2" \
#     --data_dir "$BASE_DATA_DIR/dataset-section" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/specter2-section/checkpoint-*"

# python run_clip.py \
#     --output_dir "$BASE_MODEL_DIR/specter2-meta" \
#     --model_name_or_path "$BASE_MODEL_DIR/specter2" \
#     --data_dir "$BASE_DATA_DIR/dataset-title-section" \
#     --dataset_name "../load_data_scir.py" \
#     --dataset_config_name=scir \
#     --image_column image_path \
#     --caption_column caption \
#     --remove_unused_columns=False \
#     --do_train --do_eval \
#     --per_device_train_batch_size="64" \
#     --per_device_eval_batch_size="64" \
#     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
#     --overwrite_output_dir \
#     --evaluation_strategy "steps" \

# rm -rf "$BASE_MODEL_DIR/specter2-meta/checkpoint-*"

python run_clip.py \
    --output_dir "$BASE_MODEL_DIR/specter2-meta-b192" \
    --model_name_or_path "$BASE_MODEL_DIR/specter2" \
    --data_dir "$BASE_DATA_DIR/dataset-title-section" \
    --dataset_name "../load_data_scir.py" \
    --dataset_config_name=scir \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train --do_eval \
    --per_device_train_batch_size="192" \
    --per_device_eval_batch_size="192" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \

rm -rf "$BASE_MODEL_DIR/specter2-meta-b192/checkpoint-*"

python run_clip.py \
    --output_dir "$BASE_MODEL_DIR/specter2-special-token-meta-b192" \
    --model_name_or_path "$BASE_MODEL_DIR/specter2-special-token-base" \
    --data_dir "$BASE_DATA_DIR/dataset-meta-special-token" \
    --dataset_name "../load_data_scir.py" \
    --dataset_config_name=scir \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train --do_eval \
    --per_device_train_batch_size="192" \
    --per_device_eval_batch_size="192" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --evaluation_strategy "steps" \

rm -rf "$BASE_MODEL_DIR/specter2-special-token-meta-b192/checkpoint-*"
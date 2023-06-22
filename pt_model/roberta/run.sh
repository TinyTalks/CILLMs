# Usage: bash run.sh
# 使用内置的 HuggingFaceTrainer进行训练
python path to envs/transformers/examples/pytorch/language-modeling/run_mlm.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --line_by_line \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
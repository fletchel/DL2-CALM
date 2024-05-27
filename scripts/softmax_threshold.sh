python run_summarization.py \
    --model_name_or_path ./save/cnndm_t5_small_trained/ \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config_name "3.0.0" \
    --output_dir ./save/softmax_threshold_PLACEHOLDER/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --use_early_exit True \
    --exit_conf_type softmax \
    --exit_conf_threshold PLACEHOLDER \
    --exit_min_layer 0 \
    --max_eval_samples 1000
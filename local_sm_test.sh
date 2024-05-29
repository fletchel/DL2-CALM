python run_summarization.py \
    --model_name_or_path ./save/cnndm_t5_small_run2 \
    --do_cali True \
    --dataset_name cnn_dailymail \
    --dataset_config_name "3.0.0" \
    --output_dir ./save/cali \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate True \
    --source_prefix "summarize: " \
    --max_predict_samples 500 \
    --max_calibrate_samples 50 \
    --use_early_exit True \
    --exit_conf_type softmax \
    --exit_min_layer 1 \
    --calibrate_delta .5 \
    --calibrate_epsilon 0.05 \
    --consistency_type risk \
    --thresholds 1.0 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05 \
    > outputs/output10.log 2>&1
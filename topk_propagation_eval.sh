python run_summarization.py \
    --model_name_or_path ./save/cnndm_t5_small_run2/ \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config_name "3.0.0" \
    --output_dir ./save/cnndm_t5_small_th_090_topk_2000/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --use_early_exit True \
    --exit_conf_type softmax \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 1 \
    --top_propagation 2000 \
    > outputs/output_topk_propagation.log 2>&1
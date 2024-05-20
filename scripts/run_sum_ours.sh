CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config_name "3.0.0" \
    --output_dir ./save/cnndm_t5_small_run2/ \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --save_steps 5383 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --output_hidden_states_decoder True \
    --intermediate_loss_fn weighted_ce


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 \
    run_summarization.py \
    --model_name_or_path ./save/cnndm_t5_small_run2/ \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config_name "3.0.0" \
    --output_dir ./save/cnndm_t5_small_run2/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --use_early_exit True \
    --exit_conf_type softmax \
    --exit_conf_threshold 0.9 \
    --exit_min_layer 4
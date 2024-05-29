python run_summarization.py \
  --model_name_or_path ./save/cnndm_t5_small_run2/ \
  --do_train \
  --dataset_name cnn_dailymail \
  --dataset_config_name "3.0.0" \
  --output_dir ./save/cnndm_t5_small_tmlp512_3epoch/ \
  --per_device_train_batch_size 4 \
  --overwrite_output_dir \
  --predict_with_generate \
  --source_prefix "summarize: " \
  --do_eval \
  --save_steps 100000000 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --train_meta_cm_head \
   --output_hidden_states_decoder \
  --max_eval_samples 1000 \
  --exit_conf_type transformer_MLP_512
  > outputs/output_confidence_classifier_training.log 2>&1
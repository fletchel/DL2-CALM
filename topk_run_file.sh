# T5-small CALM+topk_token_propagation with lambda=0.5 and k=20000
python run_summarization.py \
    --model_name_or_path ./save/cnndm_t5_small_run2/ \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config_name "3.0.0" \
    --output_dir ./save/cnndm_t5_small_lowdata_finetuned_v2_th_050_topk_20000/ \
    --per_device_eval_batch_size 1 \
    --deploy_scenario True \
    --use_synchronize True \
    --overwrite_output_dir \
    --predict_with_generate \
    --source_prefix "summarize: " \
    --use_early_exit True \
    --exit_conf_type softmax \
    --exit_conf_threshold 0.5 \
    --exit_min_layer 1 \
    --top_propagation 20000 \
    --max_eval_samples 100
    > outputs/output_top_k.log 2>&1
# T5-small without CALM
#python run_summarization.py \
#    --model_name_or_path ./save/cnndm_t5_small_run2/ \
#    --do_eval \
#    --dataset_name cnn_dailymail \
#    --dataset_config_name "3.0.0" \
#    --output_dir ./save/cnndm_t5_small_lowdata_finetuned_v2_baseline/ \
#    --per_device_eval_batch_size 1 \
#    --deploy_scenario True \
#    --use_synchronize True \
#    --overwrite_output_dir \
#    --predict_with_generate \
#    --source_prefix "summarize: " \
#    --use_early_exit True \
#    --exit_conf_type softmax \
#    --exit_conf_threshold 0.9 \
#    --exit_min_layer 100 \
#    --max_eval_samples 1000
#
## CALM with lambda=0.9
#python run_summarization.py \
#    --model_name_or_path ./save/cnndm_t5_small_run2/ \
#    --do_eval \
#    --dataset_name cnn_dailymail \
#    --dataset_config_name "3.0.0" \
#    --output_dir ./save/cnndm_t5_small_lowdata_finetuned_v2_th_090_no_topk/ \
#    --per_device_eval_batch_size 1 \
#    --deploy_scenario True \
#    --use_synchronize True \
#    --overwrite_output_dir \
#    --predict_with_generate \
#    --source_prefix "summarize: " \
#    --use_early_exit True \
#    --exit_conf_type softmax \
#    --exit_conf_threshold 0.9 \
#    --exit_min_layer 1 \
#    --max_eval_samples 1000
#
## CALM+topk_token_propagation with lambda=0.9 and k=2000
#python run_summarization.py \
#    --model_name_or_path ./save/cnndm_t5_small_run2/ \
#    --do_eval \
#    --dataset_name cnn_dailymail \
#    --dataset_config_name "3.0.0" \
#    --output_dir ./save/cnndm_t5_small_lowdata_finetuned_v2_th_090_topk_2000/ \
#    --per_device_eval_batch_size 1 \
#    --deploy_scenario True \
#    --use_synchronize True \
#    --overwrite_output_dir \
#    --predict_with_generate \
#    --source_prefix "summarize: " \
#    --use_early_exit True \
#    --exit_conf_type softmax \
#    --exit_conf_threshold 0.9 \
#    --exit_min_layer 1 \
#    --top_propagation 2000 \
#    --max_eval_samples 1000
#
## CALM+topk_token_propagation with lambda=0.9 and k=10000
#python run_summarization.py \
#    --model_name_or_path ./save/cnndm_t5_small_run2/ \
#    --do_eval \
#    --dataset_name cnn_dailymail \
#    --dataset_config_name "3.0.0" \
#    --output_dir ./save/cnndm_t5_small_lowdata_finetuned_v2_th_090_topk_10000/ \
#    --per_device_eval_batch_size 1 \
#    --deploy_scenario True \
#    --use_synchronize True \
#    --overwrite_output_dir \
#    --predict_with_generate \
#    --source_prefix "summarize: " \
#    --use_early_exit True \
#    --exit_conf_type softmax \
#    --exit_conf_threshold 0.9 \
#    --exit_min_layer 1 \
#    --top_propagation 10000 \
#    --max_eval_samples 1000
#
## CALM+topk_token_propagation with lambda=0.9 and k=20000
#python run_summarization.py \
#    --model_name_or_path ./save/cnndm_t5_small_run2/ \
#    --do_eval \
#    --dataset_name cnn_dailymail \
#    --dataset_config_name "3.0.0" \
#    --output_dir ./save/cnndm_t5_small_lowdata_finetuned_v2_th_090_topk_20000/ \
#    --per_device_eval_batch_size 1 \
#    --deploy_scenario True \
#    --use_synchronize True \
#    --overwrite_output_dir \
#    --predict_with_generate \
#    --source_prefix "summarize: " \
#    --use_early_exit True \
#    --exit_conf_type softmax \
#    --exit_conf_threshold 0.9 \
#    --exit_min_layer 1 \
#    --top_propagation 20000 \
#    --max_eval_samples 100
#
## CALM with lambda=0.5
#python run_summarization.py \
#    --model_name_or_path ./save/cnndm_t5_small_run2/ \
#    --do_eval \
#    --dataset_name cnn_dailymail \
#    --dataset_config_name "3.0.0" \
#    --output_dir ./save/cnndm_t5_small_lowdata_finetuned_v2_th_050_no_topk/ \
#    --per_device_eval_batch_size 1 \
#    --deploy_scenario True \
#    --use_synchronize True \
#    --overwrite_output_dir \
#    --predict_with_generate \
#    --source_prefix "summarize: " \
#    --use_early_exit True \
#    --exit_conf_type softmax \
#    --exit_conf_threshold 0.5 \
#    --exit_min_layer 1 \
#    --max_eval_samples 1000
#
## CALM+topk_token_propagation with lambda=0.5 and k=2000
#python run_summarization.py \
#    --model_name_or_path ./save/cnndm_t5_small_run2/ \
#    --do_eval \
#    --dataset_name cnn_dailymail \
#    --dataset_config_name "3.0.0" \
#    --output_dir ./save/cnndm_t5_small_lowdata_finetuned_v2_th_050_topk_2000/ \
#    --per_device_eval_batch_size 1 \
#    --deploy_scenario True \
#    --use_synchronize True \
#    --overwrite_output_dir \
#    --predict_with_generate \
#    --source_prefix "summarize: " \
#    --use_early_exit True \
#    --exit_conf_type softmax \
#    --exit_conf_threshold 0.5 \
#    --exit_min_layer 1 \
#    --top_propagation 2000 \
#    --max_eval_samples 1000
#
## CALM+topk_token_propagation with lambda=0.5 and k=10000
#python run_summarization.py \
#    --model_name_or_path ./save/cnndm_t5_small_run2/ \
#    --do_eval \
#    --dataset_name cnn_dailymail \
#    --dataset_config_name "3.0.0" \
#    --output_dir ./save/cnndm_t5_small_lowdata_finetuned_v2_th_050_topk_10000/ \
#    --per_device_eval_batch_size 1 \
#    --deploy_scenario True \
#    --use_synchronize True \
#    --overwrite_output_dir \
#    --predict_with_generate \
#    --source_prefix "summarize: " \
#    --use_early_exit True \
#    --exit_conf_type softmax \
#    --exit_conf_threshold 0.5 \
#    --exit_min_layer 1 \
#    --top_propagation 10000 \
#    --max_eval_samples 1000


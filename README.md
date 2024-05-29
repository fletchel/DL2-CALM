Code based on Fast Robust Early Exiting (https://github.com/raymin0223/fast_robust_early_exit).

First install the environment using the requirements.txt file.

Train the model by using the following command:
```bash ./scripts/train.sh```. The results from static exiting after a specific layer can be generated using ```bash ./scripts/static_layer.sh```, and the results from dynamic exiting using a softmax threshold can be generated with ```bash ./scripts/softmax_threshold.sh```.

To generate the results for the transformer and MLP models, use [LUAN SCRIPT], and to generate the result files for the top-k propagation use ```bash ./topk_run_file.sh```. 


## Calibration
In order to run the experiments for calibration please see the calibiration_run.sh script.
The important parameters are:
- '--do_cali' : this sets the calibration flag to True i.e. run the calibration.
- '--max_calibrate_samples' : the number of samples to use for calibration.
- '--exit_conf_type': '
- '--calibrate_delta': the delta value for the calibration.
- --calibrate_epsilon': the epsilon value for the calibration.
- '--thresholds': the threshold candidate for the calibration (lambda values).
- '--consistency_type': the type of consistency to use for the calibration.
For more information regarding the meaning of delta, epsilon, and consistency type, please refer to the original paper.

### Calibration Plots
In order to generate the calibration plots, please use the plot_gen_calibration.ipynb notebook. 
It contain information on how to generate the calibration plots given that you have the calibration results.

## Top-k token propagation
The evaluation of top-k token propagation method has been integrated into the standard evaluation pipeline of FREE codebase.
It can be activated by specifying the value of K for parameter ```--top_propagation K```. Since top-k propagation is defined for softmax response confidence estimation,
it takes effect only when parameter ```--exit_conf_type softmax``` is set.

The full command for evaluating with top-k token propagation is shown below:

```shell
python run_summarization.py 
    --model_name_or_path <path_to_finetuned_model> 
    --do_eval 
    --dataset_name <dataset_name>
    --dataset_config_name "3.0.0" 
    --output_dir <output_path>
    --per_device_eval_batch_size 1 
    --deploy_scenario True 
    --use_synchronize True 
    --overwrite_output_dir 
    --predict_with_generate 
    --source_prefix "summarize: " 
    --use_early_exit True 
    --exit_conf_type softmax 
    --exit_conf_threshold <confidence_threshold> 
    --exit_min_layer 1 
    --top_propagation <K>
```

As the result of this command, ```all_results.json``` file with time measurements and recorded metrics will be saved 
to ```<output_path>```.


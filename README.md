Code based on Fast Robust Early Exiting (https://github.com/raymin0223/fast_robust_early_exit).

First install the environment using the requirements.txt file.

Train the model by using the following command:
```bash ./scripts/train.sh```. The results from static exiting after a specific layer can be generated using ```bash ./scripts/static_layer.sh```, and the results from dynamic exiting using a softmax threshold can be generated with ```bash ./scripts/softmax_threshold.sh```.

To generate the results for the transformer and MLP models, use [LUAN SCRIPT], and to generate the results for the top-k propagation use [KONRAD SCRIPT]. 


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

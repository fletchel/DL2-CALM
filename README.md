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


## Classifiers

### Training
In order to train the early-exit classifiers, please see the confidence_classifier_training.sh script.
The important parameters are:
- 'do_train': sets the flag to do training
- 'output_dir': where to save the model + trained classifier
- 'learning_rate': sets learning rate for classifier training
- 'num_train_epochs': number of epochs to train for
- 'exit_conf_type': which classifier to train, options are 'vanilla_classifier' (linear), 'MLP', 'transformer_MLP_64' and 'transformer_MLP_512'. Also available are 'transformer_linear_64' and 'transformer_linear_512', which replace the MLP at the end of the transformer classifier with a simple linear layer.
- 'max_train_samples': [optional] maximum number of training datapoints to use for training

### Evaluation
In order to evaluate a trained early_exit classifier, please see the confidence_classifier_training_eval.sh script
The important parameters are:
- '--model_name_or_path': ensure this is pointed to the correct model (the one saved by the classifier training script in output_dir)
- '--use_early_exit': this must be true to ensure early exiting is used during evaluation
- '--exit_conf_type': which classifier to evaluate (note, must match the classifier trained on the model pointed to by --model_name_or_path)
- '--exit_conf_threshold': confidence threshold for use during early exiting
- '--exit_min_layer': minimum layer for early exit (in our experiments, this is always 1)
- '--max_eval_samples': [optional] maximum number of eval datapoints to use for evaluation

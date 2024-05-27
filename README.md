Code based on Fast Robust Early Exiting (https://github.com/raymin0223/fast_robust_early_exit).

First install the environment using the requirements.txt file.

Train the model by using the following command:
```bash ./scripts/train.sh```. The results from static exiting after a specific layer can be generated using ```bash ./scripts/static_layer.sh```, and the results from dynamic exiting using a softmax threshold can be generated with ```bash ./scripts/softmax_threshold.sh```.

To generate the results for the transformer and MLP models, use [LUAN SCRIPT], and to generate the results for the top-k propagation use [KONRAD SCRIPT]. Finally, to generate the calibration results, use [ANDREW + DANIEL SCRIPT].
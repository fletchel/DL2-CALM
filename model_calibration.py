import numpy as np
import logging
from transformers.trainer_utils import PredictionOutput
def calculate_empirical_average(L_values):
    return np.mean(L_values)


def hoeffding_p_value(empirical_avg, delta, n):
    return np.exp(-2 * n * (max(0, delta - empirical_avg)) ** 2)

def textual_consistency(L_early: list, L_full: list):
    '''
    L_early: output of the early-exiting model
    L_full: output of the full model
    returns: Jaccard dissimilarity between the two outputs
    '''

    # Join the lists of strings into single strings
    L_early_str = ' '.join(L_early).lower()
    L_full_str = ' '.join(L_full).lower()

    # Split the strings into lists of words
    L_early_words = L_early_str.split()
    L_full_words = L_full_str.split()

    intersection = len(set(L_early_words).intersection(set(L_full_words)))
    union = len(set(L_early_words).union(set(L_full_words)))

    return 1 - intersection / union

def calibrate(L_trainer, thresholds, delta, epsilon, cali_dataset, tokenizer,
              consistency_type='textual', logger=None):
    logger.info(f"Calibrating")
    lambda_min = 1
    L_trainer.model.set_config_exit_threshold(1)
    L_full_val: PredictionOutput = L_trainer.predict(cali_dataset, metric_key_prefix="predict")
    decoder_output_full = tokenizer.batch_decode(L_full_val.predictions, skip_special_tokens=True)

    for lambda_j in thresholds:
        L_values = []
        logger.info(f"Calibrating with threshold {lambda_j}")
        L_trainer.model.set_config_exit_threshold(lambda_j)
        L_early_val: PredictionOutput = L_trainer.predict(cali_dataset, metric_key_prefix="predict")
        decoder_output_early = tokenizer.batch_decode(L_early_val.predictions, skip_special_tokens=True)

        logger.info(f"Early (index 0): {decoder_output_early[0]}, Full: {decoder_output_full[0]}")
        logger.info(f"Early (index {len(cali_dataset)-1}): {decoder_output_early[len(cali_dataset)-1]}, Full: {decoder_output_full[len(cali_dataset)-1]}")

        if consistency_type == 'textual':
            L_val = textual_consistency(decoder_output_early, decoder_output_full)
            logger.info(f"Textual consistency: {L_val}")
        else:  # risk consistency
            L_val = max(0, L_early_val - L_full_val)

        L_values.append(L_val)

        empirical_avg = calculate_empirical_average(L_values)
        p_j = hoeffding_p_value(empirical_avg, delta, len(cali_dataset))

        # if p_j > epsilon:
        #     return lambda_j
        lambda_min = lambda_j

    return lambda_min


# Example usage
def simulate_LLMearly(index, lambda_j):
    # Simulate early model output
    return np.random.random()


def simulate_LLMfull(index):
    # Simulate full model output
    return np.random.random()



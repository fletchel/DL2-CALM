import numpy as np
import logging
from transformers.trainer_utils import PredictionOutput
from datasets import load_metric
import evaluate

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

def calibrate(trainers, thresholds, delta, epsilon, cali_dataset, tokenizer, consistency_type='textual', num_samples = 100, logger=None):
    lambda_min = 1
    num_samples = min(len(cali_dataset), num_samples) 
    cali_dataset = cali_dataset.select(range(num_samples))
    logger.info(f"Calibrating with {num_samples} samples")

    logger.info('No early exiting:')
    L_full_val: PredictionOutput = trainers[0].evaluate(cali_dataset, metric_key_prefix="eval")
    print(L_full_val)
    # decoder_output_full = tokenizer.batch_decode(L_full_val.predictions, skip_special_tokens=True) 
    # references = tokenizer.batch_decode(L_full_val.labels)

    logger.info('With early exiting:')
    for i, L_trainer in enumerate(trainers[1:]):
        L_values = []
        logger.info(f"Calibrating with trainer {i}, threshold (from model) {L_trainer.model.config.exit_conf_threshold}")

        L_early_val: PredictionOutput = L_trainer.evaluate(cali_dataset, metric_key_prefix="eval")
        print(L_early_val)
        # decoder_output_early = tokenizer.batch_decode(L_early_val.predictions, skip_special_tokens=True)

        # logger.info(f"Early (index 0): {decoder_output_early[0]}") 
        # logger.info(f"Full (index 0): {decoder_output_full[0]}")
        # logger.info(f"Early (index {num_samples-1}): {decoder_output_early[num_samples-1]}") 
        # logger.info(f"Full (index {num_samples-1}): {decoder_output_full[num_samples-1]}")

        # # TODO: the textual/risk stuff is all wrong atm
        # if consistency_type == 'textual':

        #     rouge_metric = evaluate.load("rouge")
        #     L_val = 1-rouge_metric.compute(predictions=decoder_output_early, references=decoder_output_full)["rougeL"]
        #     logger.info(f"Textual consistency: {L_val}")

        # else:  # risk consistency
            
        #     R_early = 1-rouge_metric.compute(predictions=decoder_output_early, references=references)["rougeL"]
        #     R_full = 1-rouge_metric.compute(predictions=decoder_output_full, references=references)["rougeL"]
        #     L_val = max(0, R_early - R_full)

        # L_values.append(L_val)

        empirical_avg = calculate_empirical_average(L_values)
        p_j = hoeffding_p_value(empirical_avg, delta, len(cali_dataset))

        lambda_min = i

    return lambda_min


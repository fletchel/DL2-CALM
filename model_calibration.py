import numpy as np
import logging
from transformers.trainer_utils import PredictionOutput
from datasets import load_metric
import evaluate

def calculate_empirical_average(L_values):
    return np.mean(L_values)

def hoeffding_p_value(empirical_avg, delta, n):
    return np.exp(-2 * n * (max(0, delta - empirical_avg)) ** 2)


def calibrate(trainers, thresholds, delta, epsilon, cali_dataset, tokenizer, consistency_type='textual', num_samples = 100, logger=None):
    lambda_min = 1

    num_samples = min(len(cali_dataset), num_samples) 
    cali_dataset = cali_dataset.select(range(num_samples))
    logger.info(f"Calibrating with {num_samples} samples")

    logger.info('No early exiting:')
    L_full_val: PredictionOutput = trainers[0].predict(cali_dataset, metric_key_prefix="predict")
    decoder_output_full = tokenizer.batch_decode(L_full_val.predictions, skip_special_tokens=True) 
    references = tokenizer.batch_decode(L_full_val.label_ids, skip_special_tokens=True)
    

    logger.info('With early exiting:')
    for i, L_trainer in enumerate(trainers[1:]):
        L_values = []
        logger.info(f"Calibrating with trainer {i}, threshold (from model) {L_trainer.model.config.exit_conf_threshold}")

        L_early_val: PredictionOutput = L_trainer.predict(cali_dataset, metric_key_prefix="predict")
        decoder_output_early = tokenizer.batch_decode(L_early_val.predictions, skip_special_tokens=True)

        logger.info(f"Early (index 0): {decoder_output_early[0]}") 
        logger.info(f"Full (index 0): {decoder_output_full[0]}")
        logger.info(f"Early (index {num_samples-1}): {decoder_output_early[num_samples-1]}") 
        logger.info(f"Full (index {num_samples-1}): {decoder_output_full[num_samples-1]}")

        if consistency_type == 'textual':

            rouge_metric = evaluate.load("rouge")
            L_val = 1-rouge_metric.compute(predictions=decoder_output_early, references=decoder_output_full)["rougeL"]
            logger.info(f"Textual consistency: {L_val}")

        else:  # risk consistency
            
            R_early = 1-rouge_metric.compute(predictions=decoder_output_early, references=references)["rougeL"]
            R_full = 1-rouge_metric.compute(predictions=decoder_output_full, references=references)["rougeL"]
            L_val = max(0, R_early - R_full)

        L_values.append(L_val)

        empirical_avg = calculate_empirical_average(L_values)
        p_j = hoeffding_p_value(empirical_avg, delta, len(cali_dataset))

        if p_j > epsilon:
            return lambda_min
        
        lambda_min = L_trainer.model.config.exit_conf_threshold
        

    return lambda_min


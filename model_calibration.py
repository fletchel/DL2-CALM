import numpy as np
from rouge_score import rouge_scorer

def textual_consistency(L_early: str, L_full: str):
    '''
    L_early: output of the early-exiting model
    L_full: output of the full model

    returns: dissimilarity between the two outputs, using the bounded textual consistency function rougeL
    '''

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(L_early, L_full)

    return 1 - scores['rougeL'].fmeasure 

def risk_consistency(L_early: str, L_full: str, reference: str):
    '''
    L_early: output of the early-exiting model
    L_full: output of the full model

    returns: dissimilarity between the two outputs, using a risk function derived from rougeL
    '''

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores_early = scorer.score(L_early, reference)
    scores_full = scorer.score(L_full, reference)

    return max(0, scores_full['rougeL'].fmeasure - scores_early['rougeL'].fmeasure)

def calculate_empirical_average(L_values):
    return np.mean(L_values)


def hoeffding_p_value(empirical_avg, delta, n):
    return np.exp(-2 * n * (max(0, delta - empirical_avg)) ** 2)


def calibrate(L_trainer, thresholds: list[float], delta: float, epsilon: float, samples,
              consistency_type='textual'):
    """
    Calibrate CALM for global consistency.

    Parameters:
    - L_early: Function to evaluate LLMearly(Pi, lambda_j)
    - L_full: Function to evaluate LLMfull(Pi)
    - thresholds: List of candidate thresholds in decreasing order
    - delta: Tolerance level for global consistency
    - epsilon: Tolerance for p-value under which we exit early
    - samples: List of samples to calibrate on. Each sample is a dictionary with keys 'reference' and 'candidate'
    - consistency_type: Type of consistency ('textual' or 'risk')

    Returns:
    - lambda_min: The selected threshold for early-exiting
    """
    lambda_min = 1  # Default to the most conservative threshold if none are valid
    n = len(samples)

    for lambda_j in thresholds:
        for sample in samples:
            L_values = np.array([])
            
            L_trainer.model.set_early_exit_threshold(lambda_j)
            L_early_val = L_trainer(sample['candidate'], lambda_j)
            L_trainer.model.set_early_exit_threshold(1)
            L_full_val = L_trainer(sample['candidate'], 1) 

            if consistency_type == 'textual':
                L_val = textual_consistency(L_early_val, L_full_val)

            if consistency_type == 'risk':
                L_val = risk_consistency(L_early_val, L_full_val, sample['reference']) 

            L_values.append(L_val)

        empirical_avg = calculate_empirical_average(L_values)
        p_j = hoeffding_p_value(empirical_avg, delta, n)

        if p_j > epsilon:
            return lambda_j  # Return the first threshold that meets the criterion
        lambda_min = lambda_j
    return lambda_min


# # Example usage
# def simulate_LLMearly(index, lambda_j):
#     # Simulate early model output
#     return np.random.random()


# def simulate_LLMfull(index):
#     # Simulate full model output
#     return np.random.random()

# # Define thresholds and parameters
# thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
# delta = 0.1  # Example delta
# epsilon = 0.05  # Example epsilon
# n = 100  # Number of samples in the calibration set

# selected_lambda = calibrate(simulate_LLMearly, simulate_LLMfull, thresholds, delta, epsilon, n, 'textual')
# print(f"Selected threshold for early exiting: {selected_lambda}")


# scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# scores = scorer.score('The quick brown dog jumps over the dog',
#                       'The quick brown dog jumps over the dog.')
# print(scores['rougeL'].fmeasure)

from bleurt import score

checkpoint = "bleurt/test_checkpoint"
references = ["This is a test."]
candidates = ["This is the test."]

scorer = score.BleurtScorer(checkpoint)
scores = scorer.score(references=references, candidates=candidates)
assert isinstance(scores, list) and len(scores) == 1
print(scores)


def CalibrateCALM(LLM_full, LLM_ada, sample, delta, eps, lambda_set):
    '''
    Implement the algorithm for calibrating the LLM model, following algorithm 1 of section E of the paper.

    params:
    LLM_full: the full LLM model
    LLM_ada: the early exiting LLM model. The early exiting depends on the current lambda value from lambda_set
    sample: the sample we use for calibration
    delta: tolerance parameter, how similar we want the outputs of the full and early-exit models to be
    eps: confidence rate. Since there is stochasticty from the sample, we can only guarantee that the outputs are similar with probability >= eps
    '''

    
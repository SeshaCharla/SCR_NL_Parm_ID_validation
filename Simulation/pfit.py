import numpy as np

def pfit(y: np.ndarray, y_hat: np.ndarray) -> float:
    """ Returns the percentage fit for the data and simulation """
    err = np.linalg.norm((y - y_hat), ord=2)
    # print("err = ", err)
    mean_err = np.linalg.norm((y - np.mean(y)), ord=2)
    # print("mean_err = ", mean_err)
    return 100 * (1 - (err / mean_err))

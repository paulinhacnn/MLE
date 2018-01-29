import time
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

x = np.linspace(0, 3500, num=3500)

# intercept (3), and std (3)
y_number_observation = 3 + 1.4 * x + np.random.normal(0, 1.5, 3500)

# estimating data
def likelihood_function(dados):
    dado_1 = dados[0]
    dado_2 = dados[1]
    std = dados[2]
    # regression linear
    n_predict = dado_1 + dado_2 * x

    log_l = -np.sum( stats.norm.logpdf(y_number_observation, loc=n_predict, scale=std) )
    return(log_l)

x_p = [1, 1, 1]

results = minimize(likelihood_function, x_p, method='nelder-mead')
print (results.x)

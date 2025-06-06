"""
This script contains the Wang et al. original model that will be the basis for the extension explored
in this thesis.
"""

import numpy as np
from numba import njit

def set_params(stressor_setting):
    """
    This function provides values for each parameter and calculates sigma and f1. 
    Input:
    - stressor_setting: Either "low", "medium", or "high" according to desired fluctuations in stressors.
    Output:
    - initial_params: Initial values for the model variables.
    - other_params: Other parameters used in the model.
    """
    # stressors
    if stressor_setting == "medium":
        sigma = 0.1
        mu, f1 = ((sigma**2) / 2), 0.0001
        initial_params = np.array([0.2, 0, 0, 0, 0.2, 0.05, 0.1, 0.05])
                               #(sigma, f1, K2, b2, a2, d2, e2, g2, c3, b3, d4, c41, c42, e5, c51, c52, K6,  f6,   b6,   c6,   K7,   g7,  b7,  c7,  a3,  h2,  b8,   c8, c9)
        other_params = np.array([sigma, f1, 0.2, 5, 1, 0.2, 3, 0.5, 3, 2.5, 1, 100, 0.4, 1.2, 40, 0.35, 0.1, 0.5, 0.41, 0.82, 0.05, 0.5, 0.65, 1.3, 0.3, 0.3, 0.9, 0.3, 1.0])
    if stressor_setting == "high":
        sigma = 0.12
        mu, f1 = ((sigma**2) / 2), 0.0001
        initial_params = np.array([0.2, 0, 0, 0, 0.2, 0.05, 0.1, 0.05])
                               #(sigma, f1, K2, b2, a2,d2, e2, g2, c3,  b3, d4,c41, c42, e5, c51,c52, K6,  f6,   b6,   c6,   K7,  g7,   b7,  c7,  a3,  h2,  b8,   c8, c9)
        other_params = np.array([sigma, f1, 0.2, 4, 2, 1.5, 1, 0.5, 3, 1.5, 1, 100, 0.25, 3, 50, 0.2, 0.1, 0.5, 0.41, 0.82, 0.05, 0.5, 0.65, 1.3, 0.3, 0.3, 0.9, 0.3, 0.8])
        
    return initial_params, other_params

@njit
def gen_suicide_step(current_state, dt, other_params, noise, neighbor_burden, social_connectedness, clustering_coeff):
    """
    This function simulates each step of the suicidal ideation model. 
    Input: 
    - current_state: previous state of the variable at hand.
    - dt: time step change for numerical solving. 
    - other_params: additional parameters that are not initial states.
    - noise: level of stochasticity. 
    - neighbor_burden: calculated from the graph; the sum of the neighbors' aversive internal states.
    - social_connectedness: the level of reliance and connection quality between friends.
    - clustering_coeff: calculated from the graph for each node in network_suicide.py. 
    """
    (sigma, f1, K2, b2, a2, d2, e2, g2, c3, b3, d4, c41, c42,
     e5, c51, c52, K6, f6, b6, c6, K7, g7, b7, c7, a3, h2, b8, c8, c9) = other_params

    S, A, U, T, O, E, I, SB = current_state
    mu = (sigma ** 2) / 2

    next_state = np.zeros(8)

    # Update each variable manually
    next_state[0] = max(S * np.exp((mu - (sigma**2)/2) * dt + sigma * noise - f1 * E), 0)
    next_state[1] = max(A + dt * (b2 * A * (K2 - A) + a2 * S - d2 * T - e2 * O - g2 * I + a3 * SB + (SB==0) - (clustering_coeff)), 0)
    next_state[2] = max(U + dt * (-c3 * U + b3 * A - c9 * social_connectedness), 0)
    next_state[3] = max(T + dt * (-d4 * T + (1 / (1 + np.exp(-c41 * (U - c42))))), 0)
    next_state[4] = max(O + dt * (-e5 * O + (1 / (1 + np.exp(-c51 * (U - c52))))), 0)
    next_state[5] = max(E + dt * (f6 * E * (K6 - E) + b6 * A - c6 * U), 0)
    next_state[6] = max(I + dt * (g7 * I * (K7 - I) + b7 * A - c7 * U), 0)
    if neighbor_burden == 0: 
        next_state[7] = 0
    else:
        next_state[7] = max(SB + dt * (h2 * neighbor_burden - b8 * I - c8 * SB), 0)

    return next_state

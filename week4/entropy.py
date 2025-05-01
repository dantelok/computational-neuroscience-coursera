import numpy as np


def bernoulli_entropy(p):
    """
    Suppose that we have a neuron which, in a given time period, will fire with probability 0.1, yielding a Bernoulli
    distribution for the neuron's firing (denoted by the random variable F = 0 or 1) with P(F = 1) = 0.1.

    Which of these is closest to the entropy H(F) of this distribution (calculated in bits, i.e., using the base 2
    logarithm)?
    ----------------------------------------------------------------------------------------------------
    Calculate the entropy H(F) of a Bernoulli random variable with firing probability p.

    Args:
        p (float): Probability of firing (P(F=1)), must be between 0 and 1.

    Returns:
        float: Entropy in bits.
    """
    if p <= 0 or p >= 1:
        return 0.0  # Entropy is zero when p is 0 or 1 (certain outcome)

    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return entropy


def mutual_information_flash_firing(p_flash, p_fire_given_flash, p_fire_given_no_flash):
    """
    Now let's add a stimulus to the picture.
    Suppose that we think this neuron's activity is related to a light flashing in the eye.
    Let us say that the light is flashing in a given time period with probability 0.10.
    Call this stimulus random variable S.

    If there is a flash, the neuron will fire with probability 1/2.
    If there is not a flash, the neuron will fire with probability 1/18.
    Call this random variable F (whether the neuron fires or not).

    Which of these is closest, in bits (log base 2 units), to the mutual information MI(S,F)?
    ----------------------------------------------------------------------------------------------------
    Calculate the mutual information MI(S, F) between stimulus (flash) and firing.

    Args:
        p_flash (float): Probability of a flash (P(S=1)).
        p_fire_given_flash (float): Probability of firing given flash (P(F=1|S=1)).
        p_fire_given_no_flash (float): Probability of firing given no flash (P(F=1|S=0)).

    Returns:
        float: Mutual information in bits.
    """

    p_no_flash = 1 - p_flash
    p_no_fire_given_flash = 1 - p_fire_given_flash
    p_no_fire_given_no_flash = 1 - p_fire_given_no_flash

    # Marginal probabilities
    p_fire = p_fire_given_flash * p_flash + p_fire_given_no_flash * p_no_flash
    p_no_fire = 1 - p_fire

    H_F = bernoulli_entropy(p_fire)

    # Conditional entropy H(F|S=1)
    H_F_given_S1 = bernoulli_entropy(p_fire_given_flash)

    # Conditional entropy H(F|S=0)
    H_F_given_S0 = bernoulli_entropy(p_fire_given_no_flash)

    # Total conditional entropy H(F|S)
    H_F_given_S = p_flash * H_F_given_S1 + p_no_flash * H_F_given_S0

    # Mutual Information
    MI = H_F - H_F_given_S
    return MI


print(bernoulli_entropy(0.1))
p_flash = 0.1
p_fire_given_flash = 0.5
p_fire_given_no_flash = 1/18

mi_value = mutual_information_flash_firing(p_flash, p_fire_given_flash, p_fire_given_no_flash)
print(mi_value)

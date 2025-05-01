import pickle
import numpy as np


def decode_population_vector(pop_coding_data, tuning_data):
    """
    Decode the population vector given neuron responses and basis vectors.

    Args:
        pop_coding_data (dict): Contains r1, r2, r3, r4 (responses) and c1, c2, c3, c4 (preferred direction vectors).
        tuning_data (dict): Contains neuron1, neuron2, neuron3, neuron4 firing rates over stimuli.

    Returns:
        float: Decoded stimulus direction in degrees (rounded to nearest integer, between 0 and 360).
    """
    # Calculate r_max for each neuron from the tuning data
    r_max = {
        'r1': np.max(tuning_data['neuron1']),
        'r2': np.max(tuning_data['neuron2']),
        'r3': np.max(tuning_data['neuron3']),
        'r4': np.max(tuning_data['neuron4']),
    }

    # Calculate average response for each neuron from the mystery stimulus
    r_avg = {
        'r1': np.mean(pop_coding_data['r1']),
        'r2': np.mean(pop_coding_data['r2']),
        'r3': np.mean(pop_coding_data['r3']),
        'r4': np.mean(pop_coding_data['r4']),
    }

    # Initialize population vector
    pop_vector = np.zeros(2)

    # Build the population vector by summing contributions from each neuron
    for i in range(1, 5):
        neuron_response = r_avg[f'r{i}']
        neuron_rmax = r_max[f'r{i}']
        neuron_c = pop_coding_data[f'c{i}']

        if neuron_rmax > 0:  # To avoid division by zero
            response_norm = neuron_response / neuron_rmax
            pop_vector += response_norm * neuron_c

    # According to convention: 0 degrees = positive y-axis, 90 degrees = positive x-axis
    x, y = pop_vector[0], pop_vector[1]

    # Calculate angle
    angle_rad = np.arctan2(x, y)  # Note: swap x and y
    angle_deg = np.degrees(angle_rad)

    # Normalize to [0, 360)
    angle_deg = angle_deg % 360

    return int(np.round(angle_deg))


"""
Question:
pop_coding contains four vectors named r1, r2, r3, and r4 that contain the responses (firing rate in Hz) of the four neurons to this mystery stimulus. 
It also contains four vectors named c1, c2, c3, and c4. 
These are the basis vectors corresponding to neuron 1, neuron 2, neuron 3, and neuron 4.

Decode the neural responses and recover the mystery stimulus vector by computing the population vector for these neurons. 
You should use the maximum average firing rate (over any of the stimulus values in 'tuning.mat') for a neuron as the value of r_max for that neuron. 
That is, r_max should be the maximum value in the tuning curve for that neuron.

What is the direction, in degrees, of the population vector? 
You should round your answer to the nearest degree. 
Your answer should contain the value only (no units!) and should be between 0 to 360 degrees. 
If your calculations give a negative number or a number greater than or equal to 360, 
convert it to a number in the proper range (you may use the mod function to do this).

You may need to convert your resulting vector from Cartesian coordinates to polar coordinates to find the angle. 
You may use the atan() function in MATLAB to do this. 
Note that the the convention we're using defines 0 degrees to point in the direction of the positive y-axis, 
and 90 degrees to point in the direction of the positive x-axis 
(i.e., 0 degrees is north, 90 degrees is east).
"""
# Load the data
with open('tuning_3.4.pickle', 'rb') as f:
    tuning_data = pickle.load(f)

with open('pop_coding_3.4.pickle', 'rb') as f:
    pop_coding = pickle.load(f)

# Decode the population vector
decoded_angle = decode_population_vector(pop_coding, tuning_data)
print(f"Decoded stimulus direction: {decoded_angle} degrees")

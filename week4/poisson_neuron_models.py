import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_neuron_tuning(neuron_data, stim_values, neuron_name="Neuron"):
    """
    Plot the tuning curve of a neuron given its firing rate data and stimulus values.

    Args:
        neuron_data (np.ndarray): Array of shape (n_trials, n_stimuli) with firing rates.
        stim_values (np.ndarray): Array of shape (n_stimuli,) containing stimulus directions.
        neuron_name (str): Optional name of the neuron for labeling the plot.
    """
    # Compute mean and standard deviation across trials for each stimulus
    mean_firing_rate = np.mean(neuron_data, axis=0)
    std_firing_rate = np.std(neuron_data, axis=0)

    # Plot the tuning curve with error bars
    plt.figure(figsize=(8, 5))
    plt.errorbar(stim_values, mean_firing_rate, yerr=std_firing_rate, fmt='-o', capsize=4)
    plt.title(f"Tuning Curve for {neuron_name}")
    plt.xlabel("Stimulus Direction (degrees)")
    plt.ylabel("Mean Firing Rate (Hz)")
    plt.xticks(np.arange(0, 360, 30))
    plt.grid(True)
    plt.show()


def compute_fano_factor(neuron_data):
    """
    Compute the Fano factor for each stimulus direction for a given neuron's firing rates.

    Args:
        neuron_data (np.ndarray): Shape (n_trials, n_stimuli), firing rates.

    Returns:
        np.ndarray: Fano factors for each stimulus direction.
    """
    mean_firing = np.mean(neuron_data, axis=0)  # Mean firing rate over trials
    var_firing = np.var(neuron_data, axis=0)  # Variance of firing rate over trials

    # To avoid division by zero
    fano_factors = np.zeros_like(mean_firing)
    nonzero_mask = mean_firing > 0
    fano_factors[nonzero_mask] = var_firing[nonzero_mask] / mean_firing[nonzero_mask]

    return fano_factors


"""
Question: 
The matrices contain the results of running a set of experiments in which we probed the synthetic neuron with the stimuli in stim. 
Each column of a neuron matrix contains the firing rate of that neuron (in Hz) in response to the corresponding stimulus value in stims. 
That is, n-th column of neuron1 contains the 100 trials in which we applied the stimulus of value stim(n) to neuron1. 

Plot the tuning curve -- the mean firing rate of the neuron as a function of the stimulus -- for each of the neurons.
"""
# Load the data
with open('tuning_3.4.pickle', 'rb') as f:
    data = pickle.load(f)

# Plot the graphs for each neuron
plot_neuron_tuning(data['neuron1'], data['stim'], neuron_name="Neuron 1")
plot_neuron_tuning(data['neuron2'], data['stim'], neuron_name="Neuron 2")
plot_neuron_tuning(data['neuron3'], data['stim'], neuron_name="Neuron 3")
plot_neuron_tuning(data['neuron4'], data['stim'], neuron_name="Neuron 4")

# Compute Fano factors for each neuron
fano_factors = {}
for neuron_label in ['neuron1', 'neuron2', 'neuron3', 'neuron4']:
    fano_factors[neuron_label] = compute_fano_factor(data[neuron_label])


"""
Question:
We have reason to suspect that one of the neurons is not like the others. 
Three of the neurons are Poisson neurons (they are accurately modeling using a Poisson process), 
but we believe that the remaining one might not be.

Which of the neurons (if any) is NOT Poisson?

Hint:
Note that we give you the firing rate of each of the neurons, not the spike count. 
You may find it useful to convert the firing rates to spike counts in order to test for "Poisson-ness", 
however this is not necessary.

What might this imply about the Poisson statistics (like the Fano factor) 
when we convert the spike counts (the raw output of the Poisson spike generator) into a firing rate (what we gave you)?
"""
# Plot Fano factors for all neurons
plt.figure(figsize=(10, 6))
for neuron_label, ff in fano_factors.items():
    plt.plot(data['stim'], ff, '-o', label=neuron_label)

plt.axhline(1, color='gray', linestyle='--', label='Ideal Poisson (Fano=1)')
plt.title('Fano Factor Across Stimuli')
plt.xlabel('Stimulus Direction (degrees)')
plt.ylabel('Fano Factor')
plt.legend()
plt.grid(True)
plt.show()

# Also print mean Fano factor for each neuron
for neuron_label, ff in fano_factors.items():
    print(f"{neuron_label}: Mean Fano Factor = {np.mean(ff):.3f}")

# Plot mean firing rate vs variance of firing rate for each neuron
plt.figure(figsize=(10, 6))

for neuron_label in ['neuron1', 'neuron2', 'neuron3', 'neuron4']:
    neuron_data = data[neuron_label]
    mean_firing = np.mean(neuron_data, axis=0)
    var_firing = np.var(neuron_data, axis=0)
    plt.scatter(mean_firing, var_firing, label=neuron_label)

# Plot the line variance = mean (ideal Poisson behavior)
x = np.linspace(0, np.max([np.mean(data[neuron], axis=0).max() for neuron in data if neuron.startswith('neuron')]), 100)
plt.plot(x, x, 'k--', label='Variance = Mean (Ideal Poisson)')

plt.title('Mean vs Variance of Firing Rate')
plt.xlabel('Mean Firing Rate (Hz)')
plt.ylabel('Variance of Firing Rate')
plt.legend()
plt.grid(True)
plt.show()

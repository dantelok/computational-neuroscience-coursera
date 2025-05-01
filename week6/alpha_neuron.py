"""
Created on Wed Apr 22 16:13:18 2015

Fire a neuron via alpha function synapse and random input spike train
R Rao 2007

translated to python by rkp 2015
"""
from __future__ import print_function, division

import time
import numpy as np
from numpy import concatenate as cc
import matplotlib.pyplot as plt

np.random.seed(0)
# I & F implementation dV/dt = - V/RC + I/C
h = 1.  # step size, Euler method, = dt ms
t_max = 200  # ms, simulation time period
tstop = int(t_max / h)  # number of time steps
ref = 0  # refractory period counter

# Generate random input spikes
# Note: This is not entirely realistic - no refractory period
# Also: if you change step size h, input spike train changes too...
thr = 0.9  # threshold for random spikes
spike_train = np.random.rand(tstop) > thr

# alpha func synaptic conductance
t_a = 100  # Max duration of syn conductance
t_peak = 1  # ms
g_peak = 0.05  # nS (peak synaptic conductance)
const = g_peak / (t_peak * np.exp(-1));
t_vec = np.arange(0, t_a + h, h)
alpha_func = const * t_vec * (np.exp(-t_vec / t_peak))

plt.plot(t_vec[:80], alpha_func[:80])
plt.xlabel('t (in ms)')
plt.title('Alpha Function (Synaptic Conductance for Spike at t=0)')
plt.draw()
time.sleep(2)

# capacitance and leak resistance
C = 0.5  # nF
R = 40  # M ohms
print('C = {}'.format(C))
print('R = {}'.format(R))

# conductance and associated parameters to simulate spike rate adaptation
g_ad = 0
G_inc = 1 / h
tau_ad = 2

# Initialize basic parameters
E_leak = -60  # mV, equilibrium potential
E_syn = 0  # Excitatory synapse (why is this excitatory?)
g_syn = 0  # Current syn conductance
V_th = -40  # spike threshold mV
V_spike = 50  # spike value mV
ref_max = 4 / h  # Starting value of ref period counter
t_list = np.array([], dtype=int)
V = E_leak
V_trace = [V]
t_trace = [0]

# fig, axs = plt.subplots(2, 1)
# axs[0].plot(np.arange(0, t_max, h), spike_train)
# axs[0].set_title('Input spike train')

for t in range(tstop):

    # Compute input
    if spike_train[t]:  # check for input spike
        t_list = cc([t_list, [1]])

    # Calculate synaptic current due to current and past input spikes
    g_syn = np.sum(alpha_func[t_list])
    I_syn = g_syn * (E_syn - V)

    # Update spike times
    if np.any(t_list):
        t_list = t_list + 1
        if t_list[0] == t_a:  # Reached max duration of syn conductance
            t_list = t_list[1:]

    # Compute membrane voltage
    # Euler method: V(t+h) = V(t) + h*dV/dt
    if not ref:
        V = V + h * (-((V - E_leak) * (1 + R * g_ad) / (R * C)) + (I_syn / C))
        g_ad = g_ad + h * (-g_ad / tau_ad)  # spike rate adaptation
    else:
        ref -= 1
        V = V_th - 10  # reset voltage after spike
        g_ad = 0

    # Generate spike
    if (V > V_th) and not ref:
        V = V_spike
        ref = ref_max
        g_ad = g_ad + G_inc

    V_trace += [V]
    t_trace += [t * h]

# axs[1].plot(t_trace, V_trace)
# plt.draw()
# axs[1].set_title('Output spike train')
# plt.show()


# Simulate for different t_peak, from 0.5 to 10 ms, to see the relationship between t_peak_values and firing rate
def simulate_spiking_neuron(t_peak_values, seed=0):
    """
    Simulates the spiking neuron for varying t_peak values and records output spike counts.

    Args:
        t_peak_values (list or np.ndarray): Array of t_peak values to simulate.
        seed (int): Random seed for reproducibility.

    Returns:
        list: Spike counts corresponding to each t_peak.
    """
    np.random.seed(seed)
    h = 1.  # step size
    t_max = 200  # ms
    tstop = int(t_max / h)
    ref_max = 4 / h
    thr = 0.9  # spike threshold for input
    spike_train = np.random.rand(tstop) > thr

    C = 0.5  # nF
    R = 40  # M ohms
    E_leak = -60  # mV
    E_syn = 0  # mV
    V_th = -40  # mV
    V_spike = 50  # mV

    spike_counts = []

    for t_peak in t_peak_values:
        t_a = 100  # Max duration of syn conductance
        g_peak = 0.05  # nS
        const = g_peak / (t_peak * np.exp(-1))
        t_vec = np.arange(0, t_a + h, h)
        alpha_func = const * t_vec * (np.exp(-t_vec / t_peak))

        V = E_leak
        g_ad = 0
        tau_ad = 2
        G_inc = 1 / h
        ref = 0
        t_list = np.array([], dtype=int)
        spike_count = 0

        for t in range(tstop):

            if spike_train[t]:
                t_list = cc([t_list, [1]])

            g_syn = np.sum(alpha_func[t_list])
            I_syn = g_syn * (E_syn - V)

            if np.any(t_list):
                t_list = t_list + 1
                if t_list[0] == t_a:
                    t_list = t_list[1:]

            if not ref:
                V = V + h * (-((V - E_leak) * (1 + R * g_ad) / (R * C)) + (I_syn / C))
                g_ad = g_ad + h * (-g_ad / tau_ad)
            else:
                ref -= 1
                V = V_th - 10
                g_ad = 0

            if (V > V_th) and not ref:
                V = V_spike
                ref = ref_max
                g_ad = g_ad + G_inc
                spike_count += 1

        spike_counts.append(spike_count)

    return spike_counts


# Define t_peak values from 0.5 ms to 10 ms in steps of 0.5 ms
t_peak_values = np.arange(0.5, 10.5, 0.5)
spike_counts = simulate_spiking_neuron(t_peak_values)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(t_peak_values, spike_counts, '-o')
plt.xlabel('t_peak (ms)')
plt.ylabel('Output Spike Count')
plt.title('Spike Count vs t_peak')
plt.grid(True)
plt.show()

print(f'Firing rate: {spike_counts}')


# How would you turn this synapse into an inhibitory synapse?
def simulate_spiking_neuron_Esyn_cases(t_peak_values, E_syn_cases, seed=0):
    """
    Simulates spiking neuron for different E_syn values and t_peak values.

    Args:
        t_peak_values (list or np.ndarray): Array of t_peak values to simulate.
        E_syn_cases (dict): Dictionary of {case_name: E_syn_value}.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Mapping each case_name to a list of spike counts over t_peak values.
    """
    results = {}

    for case_name, E_syn_value in E_syn_cases.items():
        np.random.seed(seed)  # Reset seed for fairness between cases
        h = 1.0
        t_max = 200  # ms
        tstop = int(t_max / h)
        ref_max = 4 / h
        thr = 0.9
        spike_train = np.random.rand(tstop) > thr

        C = 0.5  # nF
        R = 40   # M ohms
        E_leak = -60  # mV
        V_th = -40    # mV
        V_spike = 50  # mV

        spike_counts = []

        for t_peak in t_peak_values:
            t_a = 100
            g_peak = 0.05
            const = g_peak / (t_peak * np.exp(-1))
            t_vec = np.arange(0, t_a + h, h)
            alpha_func = const * t_vec * (np.exp(-t_vec / t_peak))

            V = E_leak
            g_ad = 0
            tau_ad = 2
            G_inc = 1 / h
            ref = 0
            t_list = np.array([], dtype=int)
            spike_count = 0

            for t in range(tstop):
                if spike_train[t]:
                    t_list = cc([t_list, [1]])

                g_syn = np.sum(alpha_func[t_list])
                I_syn = g_syn * (E_syn_value - V)

                if np.any(t_list):
                    t_list = t_list + 1
                    if t_list[0] == t_a:
                        t_list = t_list[1:]

                if not ref:
                    V = V + h * (-((V - E_leak) * (1 + R * g_ad) / (R * C)) + (I_syn / C))
                    g_ad = g_ad + h * (-g_ad / tau_ad)
                else:
                    ref -= 1
                    V = V_th - 10
                    g_ad = 0

                if (V > V_th) and not ref:
                    V = V_spike
                    ref = ref_max
                    g_ad = g_ad + G_inc
                    spike_count += 1

            spike_counts.append(spike_count)

        results[case_name] = spike_counts

    return results


# Set the 4 cases
E_syn_cases = {
    "E_syn < 0": -70,
    "E_syn > 0": 10,
    "E_syn < E_leak": -65,
    "E_syn < V_th": -45
}

t_peak_values = np.arange(0.5, 10.5, 0.5)
spike_count_results = simulate_spiking_neuron_Esyn_cases(t_peak_values, E_syn_cases)

# Plot the results
plt.figure(figsize=(10, 6))
for case, counts in spike_count_results.items():
    plt.plot(t_peak_values, counts, '-o', label=case)

plt.xlabel('t_peak (ms)')
plt.ylabel('Output Spike Count')
plt.title('Spike Count vs t_peak for Different E_syn Cases')
plt.grid(True)
plt.legend()
plt.show()

print(spike_count_results)

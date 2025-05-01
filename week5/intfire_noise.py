import numpy as np
import matplotlib.pyplot as plt


def simulate_interspike_intervals(noiseamp, tstop=5000):
    # Parameters
    I_base = 1  # base input current in nA
    C = 1  # capacitance in nF
    R = 40  # resistance in M ohms
    V_th = 10  # spike threshold in mV
    abs_ref = 5  # absolute refractory period (ms)
    h = 1  # timestep in ms

    # Initialize
    V = 0
    ref = 0
    V_trace = []
    spiketimes = []

    # Create noisy input
    I = I_base + noiseamp * np.random.normal(0, 1, (tstop,))

    for t in range(tstop):
        if not ref:
            V = V - (V / (R * C)) + (I[t] / C)
        else:
            ref -= 1
            V = 0.2 * V_th  # reset voltage

        if V > V_th:
            V = 50  # emit spike
            ref = abs_ref
            spiketimes.append(t)

        V_trace.append(V)

    # Calculate interspike intervals
    spiketimes = np.array(spiketimes)
    ISIs = np.diff(spiketimes)

    return ISIs


# Simulate for different noise amplitudes
noise_amps = [0, 1, 2, 3, 4, 5]
plt.figure(figsize=(15, 10))

for idx, noiseamp in enumerate(noise_amps):
    ISIs = simulate_interspike_intervals(noiseamp)

    plt.subplot(2, 3, idx + 1)
    plt.hist(ISIs, bins=30, density=True, edgecolor='black')
    plt.title(f'Noise Amplitude = {noiseamp} nA')
    plt.xlabel('Interspike Interval (ms)')
    plt.ylabel('Probability')
    plt.grid(True)

plt.tight_layout()
plt.show()

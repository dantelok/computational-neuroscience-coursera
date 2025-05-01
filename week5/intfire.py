import numpy as np
import matplotlib.pyplot as plt


def find_threshold_and_max_firing_rate(tstop=200, C=1, R=40, V_th=10, abs_ref=5, h=1):
    currents = np.arange(0, 50000, 10)  # pA, sweeping from 0 to 500 pA
    spike_counts = []

    threshold_current = None
    spike_timings_at_max = []

    for I in currents:
        I_nA = I * 1e-3  # convert pA to nA
        V = 0
        ref = 0
        spikes = 0
        spiketimes = []

        for t in range(tstop):
            if not ref:
                V = V - (V / (R * C)) + (I_nA / C)
            else:
                ref -= 1
                V = 0.2 * V_th

            if V > V_th:
                V = 50
                ref = abs_ref
                spikes += 1
                spiketimes.append(t)

        spike_counts.append(spikes)

        if threshold_current is None and spikes > 0:
            threshold_current = I

    max_spikes = max(spike_counts)
    max_index = np.argmax(spike_counts)
    best_current = currents[max_index]

    # Re-run simulation at best current to get actual spike times
    I_nA = best_current * 1e-3
    V = 0
    ref = 0
    spiketimes = []

    for t in range(tstop):
        if not ref:
            V = V - (V / (R * C)) + (I_nA / C)
        else:
            ref -= 1
            V = 0.2 * V_th

        if V > V_th:
            V = 50
            ref = abs_ref
            spiketimes.append(t)

    spiketimes = np.array(spiketimes)
    if len(spiketimes) > 1:
        isis = np.diff(spiketimes)
        mean_isi = np.mean(isis)
        max_firing_rate = int(round(1000 / mean_isi))  # Hz
    else:
        max_firing_rate = 0

    return threshold_current - 10, max_firing_rate, currents, spike_counts, spiketimes


threshold_current, max_firing_rate, currents, spike_counts, spiketimes = find_threshold_and_max_firing_rate()

# Plot spike count vs input current
plt.figure(figsize=(8, 5))
plt.plot(currents, spike_counts, marker='o')
plt.xlabel('Input Current (pA)')
plt.ylabel('Spike Count in 200 ms')
plt.title('Spike Count vs Input Current')
plt.grid(True)
plt.show()

print(threshold_current, max_firing_rate)

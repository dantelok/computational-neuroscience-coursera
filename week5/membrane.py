import numpy as np
import matplotlib.pyplot as plt


def simulate_membrane_response(R, C, I, tstop=150, h=0.2, turn_off_current=True):
    tau_theoretical = R * C
    V_inf = I * R
    V = 0
    V_trace = [V]
    times = [0]

    for t in np.arange(h, tstop, h):
        V = V + h * (-(V / (R * C)) + (I / C))
        if turn_off_current and t >= 0.6 * tstop:
            I = 0
        V_trace.append(V)
        times.append(t)

    return np.array(times), np.array(V_trace), tau_theoretical, V_inf

# Original parameters
R_original = 100  # M ohms
C_original = 0.1  # nF
I = 10  # nA
tstop = 150  # ms
h = 0.2  # ms

# Modified parameters
R_modified = 10 * R_original
C_modified = C_original / 10

# Simulate original and modified
times_orig, V_orig, tau_orig, V_inf_orig = simulate_membrane_response(R_original, C_original, I, tstop, h, turn_off_current=False)
times_mod, V_mod, tau_mod, V_inf_mod = simulate_membrane_response(R_modified, C_modified, I, tstop, h, turn_off_current=False)

print('Original V: ', V_inf_orig)
print('Modified V: ', V_inf_mod)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(times_orig, V_orig, label=f'Original (τ = {tau_orig:.1f} ms)')
plt.plot(times_mod, V_mod, linestyle='--', label=f'Modified (τ = {tau_mod:.1f} ms)')

# Add vertical lines at t = tau
plt.axvline(tau_orig, color='blue', linestyle=':', label=f'τ (Original) at {tau_orig:.1f} ms')
plt.axvline(tau_mod, color='red', linestyle=':', label=f'τ (Modified) at {tau_mod:.1f} ms')

# Mark 63.2% of steady state voltage
V_target = 0.6321 * I * R_original
plt.axhline(V_target, color='green', linestyle='--', label=f'63.2% of V_inf (~{V_target:.1f} mV)')

plt.xlabel('Time (ms)')
plt.ylabel('Membrane Voltage (mV)')
plt.title('Membrane Charging Curves with Time Constant Markers\nOriginal vs Modified (R ×10, C ÷10)')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Simulate Hodgkin-Huxley-like m_inf, h_inf, tau_m, tau_h over membrane voltage V

V = np.linspace(-100, 50, 500)  # Voltage range from -100mV to +50mV

# Approximate functions (biologically inspired)
m_inf = 1 / (1 + np.exp(-(V + 40) / 10))  # sigmoid for activation
h_inf = 1 / (1 + np.exp((V + 65) / 7))    # inverted sigmoid for inactivation

tau_m = 1 + 0.5 * np.exp(-(V + 40)**2 / 1000)  # fast activation time constant
tau_h = 5 + 20 * np.exp(-(V + 65)**2 / 3000)   # slower inactivation time constant

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot steady states
axs[0].plot(V, m_inf, label='m∞ (activation)')
axs[0].plot(V, h_inf, label='h∞ (inactivation)')
axs[0].set_title('Steady-State Activation (m∞) and Inactivation (h∞)')
axs[0].set_xlabel('Membrane Voltage (mV)')
axs[0].set_ylabel('Steady State Probability')
axs[0].legend()
axs[0].grid(True)

# Plot time constants
axs[1].plot(V, tau_m, label='τm (activation time constant)')
axs[1].plot(V, tau_h, label='τh (inactivation time constant)')
axs[1].set_title('Time Constants τm and τh vs Membrane Voltage')
axs[1].set_xlabel('Membrane Voltage (mV)')
axs[1].set_ylabel('Time Constant (ms)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

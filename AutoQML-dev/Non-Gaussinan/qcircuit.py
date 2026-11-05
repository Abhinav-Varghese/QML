import pennylane as qml
from pennylane import numpy as np

# 1. Initialize a photonic device (using the built-in Gaussian simulator)
dev = qml.device("default.gaussian", wires=2)

# 2. Define the quantum circuit
@qml.qnode(dev)
def photonic_circuit(alpha, r):
    # Displace the first qumode (mode 0) by magnitude alpha
    qml.Displacement(alpha, 0.0, wires=0)
    
    # Squeeze the second qumode (mode 1) by amount r
    qml.Squeezing(r, 0.0, wires=1)
    
    # Entangle both qumodes using a 50:50 Beamsplitter
    qml.Beamsplitter(np.pi/4, 0.0, wires=[0, 1])
    
    # 3. Measure the expected number of photons in each mode
    return [qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))]

# 4. Execute the circuit
alpha_val = 1.0  # Displacement magnitude
r_val = 0.5      # Squeezing parameter

expected_photons = photonic_circuit(alpha_val, r_val)

print(f"Expected photons in Mode 0: {expected_photons[0]:.4f}")
print(f"Expected photons in Mode 1: {expected_photons[1]:.4f}")
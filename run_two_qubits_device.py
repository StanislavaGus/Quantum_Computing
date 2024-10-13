
"""
Простая проверка работы двухкубитного устройства
"""

from symulator import TwoQubitSimulator
from constants import H, X

twoQubSim = TwoQubitSimulator()
twoQubSim.allocate_qubit()
twoQubSim.allocate_qubit()

twoQubSim.apply_single_qubit_gate(X, 0)

print(twoQubSim.measure(0))

from typing import List

import numpy as np

from constants import X, HN, PAULI_X, KET_0, KET_1, H, CNOT, PAULI_Z, RX
from symulator import NQubitSimulator

def teleport(symulator: NQubitSimulator):
    """
    Quantum teleporation
    """

    symulator.reset()

    # Поворот
    symulator.apply_single_qubit_gate(RX(np.pi/4), 0)

    # Применить Адамара к центральному кубиту
    symulator.apply_single_qubit_gate(H, 1)

    symulator.apply_n_qubit_gate(np.kron(np.eye(2), CNOT))
    #symulator.apply_n_gates(np.eye(2), CNOT)

    for i in range(symulator.dimension):
        print(f'State {i}: {symulator.get_qubit_state(i)}')


    symulator.apply_n_qubit_gate(np.kron(CNOT, np.eye(2)))
    #symulator.apply_n_gates(CNOT, np.eye(2))

    gate = np.kron(H, np.eye(2))
    gate = np.kron(gate, np.eye(2))
    symulator.apply_n_qubit_gate(gate)
    #symulator.apply_n_gates(H, np.eye(2), np.eye(2))

    # Step 5: Measure qubits 0 and 1
    measurement_results = symulator.measure_multiple_qubits([0, 1])
    print(f'Measurements: {measurement_results}')
    print('Teleport...')
    # Step 6: Apply controlled gates based on the measurement results
    # If qubit 1 was measured as 1, apply X gate to qubit 2
    symulator.controlled_by_measurement(np.eye(2), PAULI_X, measurement_results[1], 2)

    # If qubit 0 was measured as 1, apply Z gate to qubit 2
    symulator.controlled_by_measurement(np.eye(2), PAULI_Z, measurement_results[0], 2)
    print('Result...')
    # Final state of the third qubit should be the teleported state
    for i in range(symulator.dimension):
        print(f'State {i}: {symulator.get_qubit_state(i)}')

if __name__ == '__main__':
    symulator = NQubitSimulator(3)
    teleport(symulator)
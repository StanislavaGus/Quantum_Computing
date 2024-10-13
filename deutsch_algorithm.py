import numpy as np

from symulator import TwoQubitSimulator
from constants import H, X

def deutsch_algorithm(simulator: TwoQubitSimulator, oracle) -> bool:

    simulator.reset()

    simulator.apply_single_qubit_gate(X, 1) #устанавливаем первый кубит |0)

    simulator.apply_two_qubit_gate(np.kron(H, H))  # применяем адамара ко всем кубитам

    simulator.apply_two_qubit_gate(oracle)  # применяем оракула

    simulator.apply_single_qubit_gate(H, 0)

    res = simulator.measure(0)
    return "Constant" if res == 0 else "Balanced"


ORACLE1 = (
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=complex))  # f(x) = 1

ORACLE2 = (
    np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]], dtype=complex))  # f(x) = 0

ORACLE3 = (
    np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]], dtype=complex))  # f(x) = x

ORACLE4 = (
    np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 0]], dtype=complex))  # f(x) = !x



if __name__ == '__main__':

    twoQubSim = TwoQubitSimulator()

    print('ORACLE1: f(x) = 0')
    print(f'Result: {deutsch_algorithm(twoQubSim, ORACLE1)}')

    print('ORACLE2: f(x) = 1')
    print(f'Result: {deutsch_algorithm(twoQubSim, ORACLE2)}')

    print('ORACLE3: f(x) = x')
    print(f'Result: {deutsch_algorithm(twoQubSim, ORACLE3)}')

    print('ORACLE4: f(x) = !x')
    print(f'Result: {deutsch_algorithm(twoQubSim, ORACLE4)}')
import numpy as np
from symulator import NQubitSimulator
from constants import HN, CNOT
from collections import defaultdict
"""
defaultdict — это подкласс стандартного словаря (dict), который предоставляет удобный способ 
работы со значениями по умолчанию. В обычном словаре, если вы пытаетесь получить значение по 
несуществующему ключу, Python выдает ошибку KeyError. Однако defaultdict позволяет автоматически 
создавать значение по умолчанию для несуществующего ключа.
"""

def simon_algorithm(symulator: NQubitSimulator, oracle) -> list:
    """
    Алгоритм Саймона для поиска периода.
    """
    symulator.reset()

    # prepare gate H x H x H x H x I x I x I x I
    hadamard_step1_gate = HN(symulator.dimension // 2)
    for _ in range(symulator.dimension // 2):
        hadamard_step1_gate = np.kron(hadamard_step1_gate, np.eye(2))

    symulator.apply_n_qubit_gate(hadamard_step1_gate)  # Step 2
    symulator.apply_n_qubit_gate(oracle)  # Step 3
    symulator.apply_n_qubit_gate(hadamard_step1_gate)

    measured = symulator.measure_multiple_qubits(list(range(symulator.dimension // 2)))
    return measured




def example_n2_s11():
    # Example
    N = 2  # Len of s = '11'
    measured_y = set()

    # Oracle for s = 11. Taken from:
    # https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/simon.ipynb.

    oracle = CNOT(4, 0, 2) @ CNOT(4, 0, 3) @ CNOT(4, 1, 2) @ CNOT(4, 1, 3)
    ITER_COUNT = 1024
    m = defaultdict(int)

    for i in range(ITER_COUNT):
        sim = NQubitSimulator(N * 2)
        result = simon_algorithm(sim, oracle)
        if result == [0, 0]:
            continue
        measured_y.add(''.join(map(str, result)))  # Convert list to str ex: [1, 0, 1] -> '101'
        m[''.join(map(str, result))] += 1

    print(measured_y)
    print(m)
    return m['11']


def example_n3_s100():

    N = 3  # длина s
    measured_y = set()


    oracle = CNOT(6, 0, 3)
    ITER_COUNT = 1024
    m = defaultdict(int)

    for i in range(ITER_COUNT):
        sim = NQubitSimulator(N * 2)
        result = simon_algorithm(sim, oracle)
        if result == [0, 0]:
            continue
        measured_y.add(''.join(map(str, result)))  # Convert list to str ex: [1, 0, 1] -> '101'
        m[''.join(map(str, result))] += 1

    print(measured_y)
    print(m)
    return m['100']


if __name__ == '__main__':
    example_n2_s11()
    example_n3_s100()
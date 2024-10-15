import numpy as np
from symulator import NQubitSimulator
from constants import HN, CNOT_func
from collections import defaultdict

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


def oracle_s11():
    N = 2  # Длина s = '11'
    measured_y = set()

    oracle = CNOT_func(4, 0, 2) @ CNOT_func(4, 0, 3) @ CNOT_func(4, 1, 2) @ CNOT_func(4, 1, 3)
    ITER_COUNT = 1024
    m = defaultdict(int)

    print(f"Запуск алгоритма Саймона для s = '11' ({N} кубита) на {ITER_COUNT} итераций...\n")

    for i in range(ITER_COUNT):
        sim = NQubitSimulator(N * 2)
        result = simon_algorithm(sim, oracle)
        if result == [0, 0]:
            continue
        measured_str = ''.join(map(str, result))
        measured_y.add(measured_str)  # Преобразование списка в строку, например [1, 0, 1] -> '101'
        m[measured_str] += 1

    print("Измеренные уникальные результаты:")
    for key in measured_y:
        print(f"  {key}: {m[key]} раз")

    print(f"\nИтоговый результат для s = '11': {m['11']}")
    return m['11']


def oracle_s100():
    N = 3  # Длина s = '100'
    measured_y = set()

    oracle = CNOT_func(6, 0, 3)
    ITER_COUNT = 1024
    m = defaultdict(int)

    print(f"\nЗапуск алгоритма Саймона для s = '100' ({N} кубита) на {ITER_COUNT} итераций...\n")

    for i in range(ITER_COUNT):
        sim = NQubitSimulator(N * 2)
        result = simon_algorithm(sim, oracle)
        if result == [0, 0]:
            continue
        measured_str = ''.join(map(str, result))
        measured_y.add(measured_str)  # Преобразование списка в строку, например [1, 0, 1] -> '101'
        m[measured_str] += 1

    print("Измеренные уникальные результаты:")
    for key in measured_y:
        print(f"  {key}: {m[key]} раз")

    print(f"\nИтоговый результат для s = '100': {m['100']}")
    return m['100']


if __name__ == '__main__':
    oracle_s11()
    oracle_s100()

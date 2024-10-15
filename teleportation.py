from typing import List
import numpy as np
from constants import PAULI_X, H,CNOT, PAULI_Z, RX
from symulator import NQubitSimulator

def teleport(symulator: NQubitSimulator):
    """
    Квантовая телепортация
    """

    # Шаг 1: Сброс квантового симулятора в исходное состояние
    symulator.reset()
    print("\n--- Квантовая телепортация начата ---\n")

    # Шаг 2: Поворот первого кубита на угол pi/4 (подготовка состояния)
    symulator.apply_single_qubit_gate(RX(np.pi/4), 0)
    print("Применен поворот RX к первому кубиту (подготовка состояния).\n")

    # Шаг 3: Применить гейт Адамара ко второму (центральному) кубиту
    symulator.apply_single_qubit_gate(H, 1)
    print("Применен гейт Адамара ко второму кубиту.\n")

    # Шаг 4: Применить CNOT гейт между вторым и третьим кубитом (связывание)
    symulator.apply_n_qubit_gate(np.kron(np.eye(2), CNOT))
    print("Применен CNOT гейт между вторым и третьим кубитом.\n")

    # Показать состояние кубитов после CNOT
    print("Состояние кубитов после применения CNOT:\n")
    for i in range(symulator.dimension):
        print(f'Кубит {i}: {symulator.get_qubit_state(i)}')
    print("\n")

    # Шаг 5: Применить CNOT гейт между первым и вторым кубитом
    symulator.apply_n_qubit_gate(np.kron(CNOT, np.eye(2)))
    print("Применен CNOT гейт между первым и вторым кубитом.\n")

    # Применить гейт Адамара к первому кубиту
    gate = np.kron(H, np.eye(2))
    gate = np.kron(gate, np.eye(2))
    symulator.apply_n_qubit_gate(gate)
    print("Применен гейт Адамара к первому кубиту.\n")

    # Шаг 6: Измерение кубитов 0 и 1
    measurement_results = symulator.measure_multiple_qubits([0, 1])
    print(f'Результаты измерений кубитов 0 и 1: {measurement_results}')
    print("\nТелепортация...\n")

    # Шаг 7: Применение контролируемых гейтов в зависимости от результатов измерения
    # Если кубит 1 был измерен как 1, применить X гейт к кубиту 2
    symulator.controlled_by_measurement(np.eye(2), PAULI_X, measurement_results[1], 2)
    print(f"Применен X гейт к третьему кубиту на основе измерения кубита 1 ({measurement_results[1]}).\n")

    # Если кубит 0 был измерен как 1, применить Z гейт к кубиту 2
    symulator.controlled_by_measurement(np.eye(2), PAULI_Z, measurement_results[0], 2)
    print(f"Применен Z гейт к третьему кубиту на основе измерения кубита 0 ({measurement_results[0]}).\n")

    # Показать финальное состояние кубитов
    print("Результат телепортации - финальное состояние кубитов:\n")
    for i in range(symulator.dimension):
        print(f'Кубит {i}: {symulator.get_qubit_state(i)}')
    print("\n--- Телепортация завершена ---\n")

if __name__ == '__main__':
    symulator = NQubitSimulator(3)
    teleport(symulator)

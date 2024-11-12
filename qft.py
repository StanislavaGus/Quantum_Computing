import typing as t
import math
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer

# Основная функция для создания квантовой схемы QFT
def create_qft_circuit(
    initialize_state: t.Callable[[QuantumCircuit, int, t.List[int]], None],
    nqubits: int,
    initial_state: t.List[int] = None,
    reverse: bool = False
) -> QuantumCircuit:
    """Создает схему для квантового преобразования Фурье с заданным числом кубитов."""
    circuit = QuantumCircuit()
    qubits = QuantumRegister(nqubits, name='q')
    circuit.add_register(qubits)
    initialize_state(circuit, nqubits, initial_state)
    apply_qft_rotations(circuit, nqubits, reverse)
    apply_swap_gates(circuit, nqubits)
    return circuit

def apply_qft_rotations(circuit: QuantumCircuit, n: int, reverse: bool = False) -> None:
    """Добавляет вращения и операцию Адамара для каждого кубита."""
    if n == 0:
        return
    n -= 1
    qubits = circuit.qubits
    circuit.h(qubits[n])
    k = -1 if reverse else 1
    for i in range(n):
        circuit.cp(k * math.pi / 2 ** (n - i), qubits[i], qubits[n])
    apply_qft_rotations(circuit, n, reverse=reverse)

def apply_swap_gates(circuit: QuantumCircuit, n: int) -> None:
    """Переставляет кубиты так, чтобы они были выведены в нужном порядке."""
    qubits = circuit.qubits
    for i in range(n // 2):
        circuit.swap(qubits[i], qubits[n - i - 1])

def initialize_state(circuit: QuantumCircuit, n: int, initial_state: t.List[int] = None) -> None:
    """Устанавливает начальное состояние кубитов в зависимости от переданного состояния."""
    if initial_state is None:
        initial_state = [1 if i == 0 or i == n - 1 else 0 for i in range(n)]
    for i, state in enumerate(initial_state):
        if state == 1:
            circuit.x(i)

# Пример начального состояния
initial_state = [1, 0, 1]

# Создаем схему для квантового преобразования Фурье с 3 кубитами
qft_circuit = create_qft_circuit(initialize_state, nqubits=3, initial_state=initial_state)
qft_circuit.measure_all()

# Запускаем схему на симуляторе
simulator = Aer.get_backend('aer_simulator')
compiled_circuit = transpile(qft_circuit, simulator)
result = simulator.run(compiled_circuit).result()

# Получаем результаты измерений и преобразуем в таблицу
counts = result.get_counts()
results_table = pd.DataFrame(list(counts.items()), columns=["Значение", "Число измерений"])

# Печать таблицы
print(results_table)

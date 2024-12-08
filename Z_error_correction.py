import numpy as np
import matplotlib.pyplot as plt
import itertools
from symulator import NQubitSimulator
from constants import PAULI_X, PAULI_Z, CNOT, P_0, P_1, CNOT_func, KET_0, KET_1, H, P_0, P_1


def rotation_x(theta):
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)

def rotation_y(theta):
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)

def rotation_z(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)




def CNOT_func(num_qubits, control, target):
    """Создаём матрицу CNOT для заданных управляющего и целевого кубитов."""
    dim = 2 ** num_qubits
    cnot_gate = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bin_i = list(format(i, f'0{num_qubits}b'))
        # состояние i
        if bin_i[control] == '1':
            # flip target
            bin_i[target] = '1' if bin_i[target] == '0' else '0'
            j = int(''.join(bin_i), 2)
            cnot_gate[j, i] = 1.0
        else:
            cnot_gate[i, i] = 1.0
    return cnot_gate


def encode(simulator, data_qubit_indices):
    """
    Кодирование для исправления фазовых ошибок:
    1. Применяем H ко всем трём кубитам данных: теперь |0>->|+>, |1>->|->
    2. Применяем CNOT(q0->q1) и CNOT(q0->q2) как в классическом коде.
    В итоге получим логические состояния |+++> или |--->
    """

    # CNOT для репликации фазы в X-базисе (аналог классического кода)
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[1]))
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[2]))

    for q in data_qubit_indices:
        simulator.apply_single_qubit_gate(H, q)


def decode(simulator, data_qubit_indices):
    """
    Декодирование - обратные шаги:
    1. Обратные CNOT
    2. Применяем H на каждый кубит, чтобы вернуться в вычислительный базис.
    """
    for q in data_qubit_indices:
        simulator.apply_single_qubit_gate(H, q)

    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[1]))
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[2]))




def introduce_phase_errors(simulator, data_qubit_indices, p):
    """
    Внесение фазовых ошибок (Z) с вероятностью p.
    """
    for qubit in data_qubit_indices:
        if np.random.random() < p:
            simulator.apply_single_qubit_gate(PAULI_Z, qubit)


def measure_syndrome(simulator, data_qubit_indices, ancilla_qubit_indices):
    """
    Измеряем синдром, как в классическом коде, но нам нужно перейти к вычислительному базису.
    Шаги:
    1. Применяем H к каждому кубиту данных, чтобы перейти из X-базиса в Z-базис.
    2. Измеряем синдром классическим способом (CNOT от кубитов данных к ancilla).
    3. Измеряем ancilla.
    4. Применяем H к кубитам данных, чтобы вернуться в X-базис, где будем исправлять ошибку.

    ВНИМАНИЕ: В отличие от классического случая, здесь мы после измерения синдрома и определения ошибки
    должны вернуться в X-базис перед коррекцией.
    """

    # Переход в вычислительный базис:
    for q in data_qubit_indices:
        simulator.apply_single_qubit_gate(H, q)

    # Синдром s1
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], ancilla_qubit_indices[0]))
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[1], ancilla_qubit_indices[0]))

    # Синдром s2
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], ancilla_qubit_indices[1]))
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[2], ancilla_qubit_indices[1]))

    s1 = simulator.measure(ancilla_qubit_indices[0])
    s2 = simulator.measure(ancilla_qubit_indices[1])

    # Возвращаемся в X-базис для коррекции
    for q in data_qubit_indices:
        simulator.apply_single_qubit_gate(H, q)

    return (s1, s2)


def correct_errors(simulator, syndrome, data_qubit_indices):
    """
    По синдрому определяем, какой кубит ошибся.
    В фазовом коде для исправления Z-ошибки на i-том кубите мы применяем оператор Z к этому кубиту в X-базисе (текущем).
    """
    s1, s2 = syndrome

    # Расшифровка синдрома такая же, как и в классическом коде, но теперь ошибка - это фаза (Z):
    # (0,0) - нет ошибки
    # (1,0) - ошибка во втором кубите
    # (0,1) - ошибка в третьем кубите
    # (1,1) - ошибка в первом кубите
    if (s1, s2) == (1, 0):
        # Ошибка во втором кубите
        simulator.apply_single_qubit_gate(PAULI_Z, data_qubit_indices[1])
    elif (s1, s2) == (0, 1):
        simulator.apply_single_qubit_gate(PAULI_Z, data_qubit_indices[2])
    elif (s1, s2) == (1, 1):
        simulator.apply_single_qubit_gate(PAULI_Z, data_qubit_indices[0])
    # (0,0) - ничего не делаем


def simulate_correction(p, n_runs, simulator_class, n_qubits):
    """
    Симуляция с коррекцией фазовых ошибок.
    """
    data_qubit_indices = [0, 1, 2]
    ancilla_qubit_indices = [3, 4]
    error_after_correction = 0

    for _ in range(n_runs):
        simulator = simulator_class(n_qubits)

        # Кодируем логический кубит (предполагается, что начальное состояние |0>)
        encode(simulator, data_qubit_indices)

        # Вносим фазовые ошибки
        introduce_phase_errors(simulator, data_qubit_indices, p)

        # Измеряем синдром
        syndrome = measure_syndrome(simulator, data_qubit_indices, ancilla_qubit_indices)

        # Корректируем
        correct_errors(simulator, syndrome, data_qubit_indices)

        # Декодируем
        decode(simulator, data_qubit_indices)

        # Измеряем логический кубит
        decoded = simulator.measure(data_qubit_indices[0])

        # Исходное состояние было |0>, если получили 1 - ошибка
        if decoded != 0:
            error_after_correction += 1

    return error_after_correction / n_runs


def simulate_no_correction(p, n_runs, simulator_class, n_qubits):
    """
    Симуляция без коррекции:
    Кодируем, вносим ошибки Z, декодируем и измеряем, без измерения синдрома и исправления.
    """
    data_qubit_indices = [0, 1, 2]
    error_without_correction = 0
    # Без ancilla
    for _ in range(n_runs):
        simulator = simulator_class(n_qubits)
        encode(simulator, data_qubit_indices)
        introduce_phase_errors(simulator, data_qubit_indices, p)
        decode(simulator, data_qubit_indices)
        decoded = simulator.measure(data_qubit_indices[0])
        if decoded != 0:
            error_without_correction += 1

    return error_without_correction / n_runs


def theoretical_p_e(p):
    """
    Теоретическая вероятность ошибки после коррекции для трёхкубитного кода:
    p_e = 3p^2 - 2p^3
    """
    return 3 * p**2 * (1 - p) + p**3


def plot_simulation(simulated_p_e, p_values, p_e_no_correction, theoretical_p_e_values):
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, simulated_p_e, 'o-', label='Симуляция с коррекцией (фазовые ошибки)', color='blue')
    plt.plot(p_values, theoretical_p_e_values, '^-', label='Теоретическая вероятность', color='green')
    plt.plot(p_values, p_e_no_correction, 's--', label='Симуляция без коррекции', color='red')
    plt.plot(p_values, p_values, label='p_e = p (без коррекции)', color='black', linestyle=':')

    plt.title("Z-ошибки")
    plt.xlabel("Вероятность фазовой ошибки p")
    plt.ylabel("Вероятность ошибки после декодирования p_e")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    n_qubits = 5  # 3 data + 2 ancilla
    n_runs = 1000
    p_values = np.linspace(0, 0.5, 10)

    simulated_p_e = []
    p_e_no_correction = []
    theoretical_p_e_values = []

    for p in p_values:
        print(f"Simulating for p = {p:.2f}")
        p_e_corr = simulate_correction(p, n_runs, NQubitSimulator, n_qubits)
        simulated_p_e.append(p_e_corr)

        p_e_nc = simulate_no_correction(p, n_runs, NQubitSimulator, n_qubits)
        p_e_no_correction.append(p_e_nc)

        p_e_th = theoretical_p_e(p)
        theoretical_p_e_values.append(p_e_th)

        print(f"p={p:.2f}: corrected={p_e_corr:.4f}, no_correction={p_e_nc:.4f}, theory={p_e_th:.4f}")

    plot_simulation(simulated_p_e, p_values, p_e_no_correction, theoretical_p_e_values)


if __name__ == "__main__":
    main()

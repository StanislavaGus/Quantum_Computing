import numpy as np
import matplotlib.pyplot as plt
from symulator import NQubitSimulator
from constants import X, CNOT, P_0, P_1, CNOT_func
import itertools


def encode(simulator, data_qubit_indices, ancilla_qubit_indices):
    """
    Кодирование логического кубита в три физических кубита с использованием тройного повторяющегося кода.
    :param simulator: Экземпляр NQubitSimulator.
    :param data_qubit_indices: Список индексов физических кубитов, представляющих логический кубит.
    :param ancilla_qubit_indices: Список индексов ancilla кубитов для измерения синдрома.
    """
    # Пример для логического |0>:
    # Здесь предполагается, что все кубиты инициализированы в |0>.
    # Для общего состояния нужно будет расширить алгоритм.

    # Применяем CNOT от первого кубита к остальным для кодирования.
    # Это эквивалентно |0> -> |000>, |1> -> |111>.
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[1]))
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[2]))


def decode(simulator, data_qubit_indices):
    """
    Декодирование тройного повторяющегося кода обратно в логический кубит.
    :param simulator: Экземпляр NQubitSimulator.
    :param data_qubit_indices: Список индексов физических кубитов, представляющих логический кубит.
    """
    # Применяем CNOT от первого кубита к остальным для декодирования.
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[1]))
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], data_qubit_indices[2]))


def introduce_errors(simulator, data_qubit_indices, p):
    """
    Введение битовых (X) ошибок с вероятностью p к каждому физическому кубиту.
    :param simulator: Экземпляр NQubitSimulator.
    :param data_qubit_indices: Список индексов физических кубитов, представляющих логический кубит.
    :param p: Вероятность ошибки на одном кубите.
    """
    for qubit in data_qubit_indices:
        if np.random.random() < p:
            simulator.apply_single_qubit_gate(X, qubit)


def measure_syndrome(simulator, data_qubit_indices, ancilla_qubit_indices):
    """
    Измерение синдрома ошибок с использованием ancilla кубитов.
    :param simulator: Экземпляр NQubitSimulator.
    :param data_qubit_indices: Список индексов физических кубитов, представляющих логический кубит.
    :param ancilla_qubit_indices: Список индексов ancilla кубитов для измерения синдрома.
    :return: Кортеж (s1, s2) - результаты измерения синдромов.
    """
    # Синдром s1: Паритет между кубитами 0 и 1
    # Синдром s2: Паритет между кубитами 0 и 2

    # Применяем CNOTы для измерения синдромов
    # Используем ancilla кубиты как "контроллеров" для паритета

    # Синдром s1
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], ancilla_qubit_indices[0]))
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[1], ancilla_qubit_indices[0]))

    # Синдром s2
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[0], ancilla_qubit_indices[1]))
    simulator.apply_n_qubit_gate(CNOT_func(simulator.dimension, data_qubit_indices[2], ancilla_qubit_indices[1]))

    # Измеряем ancilla кубиты
    s1 = simulator.measure(ancilla_qubit_indices[0])
    s2 = simulator.measure(ancilla_qubit_indices[1])

    return (s1, s2)


def correct_errors(simulator, syndrome, data_qubit_indices):
    """
    Коррекция ошибок на основе измеренного синдрома.
    :param simulator: Экземпляр NQubitSimulator.
    :param syndrome: Кортеж (s1, s2) - результаты измерения синдромов.
    :param data_qubit_indices: Список индексов физических кубитов, представляющих логический кубит.
    """
    s1, s2 = syndrome
    # Определяем, какой кубит ошибся
    if (s1, s2) == (0, 0):
        # Нет ошибок
        pass
    elif (s1, s2) == (1, 0):
        # Ошибка во втором кубите (индекс 1)
        simulator.apply_single_qubit_gate(X, data_qubit_indices[1])
    elif (s1, s2) == (0, 1):
        # Ошибка в третьем кубите (индекс 2)
        simulator.apply_single_qubit_gate(X, data_qubit_indices[2])
    elif (s1, s2) == (1, 1):
        # Ошибка в первом кубите (индекс 0)
        simulator.apply_single_qubit_gate(X, data_qubit_indices[0])


def simulate_correction(p, n_runs, simulator_class, n_qubits):
    """
    Симуляция коррекции ошибок для заданной вероятности p.
    :param p: Вероятность ошибки на одном кубите.
    :param n_runs: Количество симуляций.
    :param simulator_class: Класс симулятора (например, NQubitSimulator).
    :param n_qubits: Общее количество кубитов в симуляторе.
    :return: Вероятность ошибки после коррекции.
    """
    error_after_correction = 0

    # Индексы кубитов:
    # 0,1,2 - данные кубиты
    # 3,4 - ancilla кубиты для измерения синдрома
    data_qubit_indices = [0, 1, 2]
    ancilla_qubit_indices = [3, 4]

    for _ in range(n_runs):
        # Инициализация симулятора
        simulator = simulator_class(n_qubits)

        # Кодирование
        encode(simulator, data_qubit_indices, ancilla_qubit_indices)

        # Введение ошибок
        introduce_errors(simulator, data_qubit_indices, p)

        # Измерение синдрома
        syndrome = measure_syndrome(simulator, data_qubit_indices, ancilla_qubit_indices)

        # Коррекция ошибок на основе синдрома
        correct_errors(simulator, syndrome, data_qubit_indices)

        # Декодирование
        decode(simulator, data_qubit_indices)

        # Измерение логического кубита (qubit 0)
        decoded = simulator.measure(0)

        # Проверка, соответствует ли декодированный кубит исходному состоянию |0>
        if decoded != 0:
            error_after_correction += 1

    # Вычисление средней вероятности ошибки после коррекции
    p_e_corrected = error_after_correction / n_runs
    return p_e_corrected


def simulate_no_correction(p, n_runs, simulator_class, n_qubits):
    """
    Симуляция без применения схемы коррекции ошибок.
    :param p: Вероятность ошибки на одном кубите.
    :param n_runs: Количество симуляций.
    :param simulator_class: Класс симулятора (например, NQubitSimulator).
    :param n_qubits: Общее количество кубитов в симуляторе.
    :return: Вероятность ошибки без коррекции.
    """
    error_without_correction = 0
    data_qubit_indices = [0, 1, 2]

    for _ in range(n_runs):
        # Инициализация симулятора
        simulator = simulator_class(n_qubits)

        # Кодирование
        encode(simulator, data_qubit_indices, ancilla_qubit_indices=[])

        # Введение ошибок
        introduce_errors(simulator, data_qubit_indices, p)

        # Декодирование
        decode(simulator, data_qubit_indices)

        # Измерение логического кубита (qubit 0)
        decoded = simulator.measure(0)

        # Проверка, соответствует ли декодированный кубит исходному состоянию |0>
        if decoded != 0:
            error_without_correction += 1

    # Вычисление средней вероятности ошибки без коррекции
    p_e_no_correction = error_without_correction / n_runs
    return p_e_no_correction


def theoretical_p_e(p):
    """Теоретическая вероятность ошибки после коррекции."""
    return 3 * p ** 2 * (1 - p) + p ** 3  # = 3p^2 - 2p^3


def plot_simulation(simulated_p_e, p_values, p_e_no_correction, theoretical_p_e_values):
    """Построение графика зависимости вероятности ошибки после коррекции от p."""
    plt.figure(figsize=(10, 6))

    # Симуляция с коррекцией ошибок
    plt.plot(p_values, simulated_p_e, 'o-', label='Симуляция с коррекцией', color='blue')

    # Теоретическая вероятность ошибки после коррекции
    plt.plot(p_values, theoretical_p_e_values, '^-', label='Теоретическая вероятность', color='green')

    # Симуляция без коррекции ошибок
    plt.plot(p_values, p_e_no_correction, 's--', label='Симуляция без коррекции', color='red')

    # Прямая линия p_e = p (вероятность без коррекции)
    plt.plot(p_values, p_values, label='Вероятность без коррекции (p_e = p)', color='black', linestyle=':')

    plt.title("Вероятность ошибки после декодирования тройным повторяющимся кодом")
    plt.xlabel("Вероятность битовой ошибки на кубите p")
    plt.ylabel("Вероятность ошибки после декодирования p_e")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Параметры симуляции
    n_qubits = 5  # 3 data qubits + 2 ancilla qubits
    n_runs = 1000  # Количество симуляций для каждого p
    p_values = np.linspace(0, 0.5, 20)  # Значения p от 0 до 0.5

    simulated_p_e = []
    p_e_no_correction = []
    theoretical_p_e_values = []

    for p in p_values:
        print(f"Simulating for p = {p:.2f}")

        # Симуляция с коррекцией ошибок
        p_e_corr = simulate_correction(p, n_runs, NQubitSimulator, n_qubits)
        simulated_p_e.append(p_e_corr)

        # Симуляция без коррекции ошибок
        p_e_nc = simulate_no_correction(p, n_runs, NQubitSimulator, n_qubits)
        p_e_no_correction.append(p_e_nc)

        # Теоретическое значение
        p_e_th = theoretical_p_e(p)
        theoretical_p_e_values.append(p_e_th)

        print(
            f"p={p:.2f}: Simulated p_e_corr={p_e_corr:.4f}, Simulated p_e_no_corr={p_e_nc:.4f}, Theoretical p_e={p_e_th:.4f}")

    # Построение графика
    plot_simulation(simulated_p_e, p_values, p_e_no_correction, theoretical_p_e_values)


if __name__ == "__main__":
    main()

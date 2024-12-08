import numpy as np
import matplotlib.pyplot as plt

# Определение операторов и состояний
PAULI_X = np.array([[0, 1], [1, 0]])  # Оператор X (битовая ошибка)
KET_0 = np.array([1, 0])  # Состояние |0>
KET_1 = np.array([0, 1])  # Состояние |1>


def apply_x(qubit):
    """Применяет оператор X к заданному состоянию."""
    return PAULI_X @ qubit


def encode(state):
    """
    Кодирование одного кубита в три кубита (тройной повторяющийся код).
    |0⟩ → |000⟩
    |1⟩ → |111⟩
    """
    if np.array_equal(state, KET_0):
        return [KET_0.copy(), KET_0.copy(), KET_0.copy()]
    elif np.array_equal(state, KET_1):
        return [KET_1.copy(), KET_1.copy(), KET_1.copy()]
    else:
        raise ValueError("Некорректное состояние для кодирования.")


def introduce_errors(encoded, p):
    """Введение битовых ошибок X к каждому кубиту с вероятностью p."""
    errored = []
    for qubit in encoded:
        if np.random.random() < p:
            errored_qubit = apply_x(qubit)
            errored.append(errored_qubit)
        else:
            errored.append(qubit.copy())
    return errored


def measure_syndrome(errored):
    """
    Измерение синдрома ошибок с использованием двух ancilla кубитов.
    Возвращает два битовых синдрома (s1, s2).
    """
    q0, q1, q2 = errored
    # Синдром s1: Паритет между q0 и q1
    # Синдром s2: Паритет между q0 и q2
    s1 = 1 if not np.array_equal(q0, q1) else 0
    s2 = 1 if not np.array_equal(q0, q2) else 0
    return (s1, s2)


def correct_errors(errored, syndrome):
    """
    Коррекция ошибок на основе синдрома.
    """
    s1, s2 = syndrome
    # Определяем, какой кубит ошибся
    if (s1, s2) == (0, 0):
        # Нет ошибок
        return errored.copy()
    elif (s1, s2) == (1, 0):
        # Ошибка в q1
        errored[1] = apply_x(errored[1])
    elif (s1, s2) == (0, 1):
        # Ошибка в q2
        errored[2] = apply_x(errored[2])
    elif (s1, s2) == (1, 1):
        # Ошибка в q0
        errored[0] = apply_x(errored[0])
    return errored.copy()


def decode(errored):
    """Декодирование трёх кубитов обратно в один логический кубит с использованием большинства."""
    count_0 = sum(np.array_equal(q, KET_0) for q in errored)
    count_1 = 3 - count_0
    return KET_1.copy() if count_1 > count_0 else KET_0.copy()


def simulate_correction(p, n_runs=10000):
    """Симуляция коррекции ошибок для заданной вероятности p."""
    error_after_correction = 0
    # Предполагаем, что логический кубит всегда начинается в состоянии |0>
    for _ in range(n_runs):
        original = KET_0.copy()
        encoded = encode(original)
        errored = introduce_errors(encoded, p)
        syndrome = measure_syndrome(errored)
        corrected = correct_errors(errored, syndrome)
        decoded = decode(corrected)
        if not np.array_equal(decoded, original):
            error_after_correction += 1
    p_e_corrected = error_after_correction / n_runs
    return p_e_corrected


def theoretical_p_e(p):
    """Теоретическая вероятность ошибки после коррекции."""
    return 3 * p ** 2 * (1 - p) + p ** 3  # = 3p^2 - 2p^3


def plot_simulation(n_runs=10000):
    """Построение графика зависимости вероятности ошибки после коррекции от p."""
    p_values = np.linspace(0, 0.5, 50)  # Ограничиваем p до 0.5
    simulated_p_e = []
    theoretical_p_e_values = []

    for p in p_values:
        p_e_corr = simulate_correction(p, n_runs)
        simulated_p_e.append(p_e_corr)
        theoretical_p_e_values.append(theoretical_p_e(p))
        print(f"p={p:.2f}: Simulated p_e_corr={p_e_corr:.4f}, Theoretical p_e={theoretical_p_e(p):.4f}")

    plt.figure(figsize=(10, 6))

    # Симуляция с коррекцией ошибок
    plt.plot(p_values, simulated_p_e, 'o-', label='Симуляция с коррекцией', color='blue')

    # Теоретическая вероятность ошибки после коррекции
    plt.plot(p_values, theoretical_p_e_values, '^-', label='Теоретическая вероятность', color='green')

    # Прямая линия p_e = p (вероятность без коррекции)
    plt.plot(p_values, p_values, label='Вероятность без коррекции', color='red', linestyle='--', linewidth=2)

    plt.title("Вероятность ошибки после коррекции тройным повторяющимся кодом")
    plt.xlabel("Вероятность битовой ошибки на кубите p")
    plt.ylabel("Вероятность ошибки после декодирования p_e")
    plt.legend()
    plt.grid(True)
    plt.show()


# Вызов функции для построения графиков
plot_simulation()

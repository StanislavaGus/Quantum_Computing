import numpy as np
import matplotlib.pyplot as plt
from constants import PAULI_X, KET_0, KET_1

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
    Измерение синдрома ошибок.
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
        pass
    elif (s1, s2) == (1, 0):
        # Ошибка во втором кубите (q1)
        errored[1] = apply_x(errored[1])
    elif (s1, s2) == (0, 1):
        # Ошибка в третьем кубите (q2)
        errored[2] = apply_x(errored[2])
    elif (s1, s2) == (1, 1):
        # Ошибка в первом кубите (q0)
        errored[0] = apply_x(errored[0])
    return errored.copy()


def decode(corrected):
    """Декодирование трёх кубитов обратно в один логический кубит на основе коррекции."""
    # После коррекции все кубиты должны быть одинаковыми
    # Проверяем соответствие кодовым словам |000> или |111>
    if np.array_equal(corrected[0], KET_0) and np.array_equal(corrected[1], KET_0) and np.array_equal(corrected[2],
                                                                                                      KET_0):
        return KET_0.copy()
    elif np.array_equal(corrected[0], KET_1) and np.array_equal(corrected[1], KET_1) and np.array_equal(corrected[2],
                                                                                                        KET_1):
        return KET_1.copy()
    else:
        # Неконсистентное состояние после коррекции
        raise ValueError("Неконсистентное состояние после коррекции.")


def simulate_correction(p, n_runs=10000):
    """Симуляция коррекции битовых ошибок с использованием тройного повторяющегося кода."""
    error_after_correction = 0
    for _ in range(n_runs):
        # Исходное состояние логического кубита (например, |0>)
        original = KET_0.copy()

        # Кодирование
        encoded = encode(original)

        # Введение ошибок
        errored = introduce_errors(encoded, p)

        # Измерение синдрома
        syndrome = measure_syndrome(errored)

        # Коррекция ошибок на основе синдрома
        corrected = correct_errors(errored, syndrome)

        # Декодирование обратно в логический кубит
        try:
            decoded = decode(corrected)
            # Проверка, отличается ли декодированный кубит от исходного
            if not np.array_equal(decoded, original):
                error_after_correction += 1
        except ValueError:
            # Неконсистентное состояние после коррекции считается ошибкой
            error_after_correction += 1

    # Вычисление средней вероятности ошибки после коррекции
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

import numpy as np
import itertools
from typing import List
from constants import X, HN
from symulator import NQubitSimulator


def bernstein_vazirani(sim: NQubitSimulator, oracle) -> List[bool]:

    sim.reset()  # Шаг 1: Сбрасываем квантовый симулятор в состояние |0...0>
    sim.apply_single_qubit_gate(X,
                                sim.dimension - 1)  # Применяем гейт X к последнему кубиту, чтобы подготовить его в |1>

    # Шаг 2: Применяем гейты Адамара ко всем кубитам
    sim.apply_n_qubit_gate(HN(sim.dimension))

    # Шаг 3: Применяем оракул для задачи Бернштейна-Вазирани
    sim.apply_n_qubit_gate(oracle)

    # Шаг 4: Применяем гейты Адамара снова ко всем кубитам, кроме последнего
    sim.apply_n_qubit_gate(HN(sim.dimension))

    # Шаг 5: Измеряем все кубиты, кроме последнего вспомогательного
    measured = []
    for i in range(sim.dimension - 1):
        measured.append(sim.measure(i))

    return measured


def generate_oracle_bernstein_vazirani(N, s):
    """
    Генерация оракула для задачи Бернштейна-Вазирани.

    Аргументы:
    - N: количество переменных.
    - s: скрытая строка, используемая для скалярного произведения в булевой функции.

    Возвращает:
    - Матрица размером 2^(N+1) x 2^(N+1), описывающая оракул.
    """
    # Размер матрицы оракула
    matrix_size = 2 ** (N + 1)

    # Инициализация нулевой матрицы
    oracle_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # Проход по всем возможным входным данным (x) и вспомогательному биту (y)
    for x in itertools.product([0, 1], repeat=N):
        for y in [0, 1]:  # y - вспомогательный кубит
            # Индекс входа (двоичное представление x, за которым следует y)
            input_index = int(''.join(map(str, x)) + str(y), 2)

            # Вычисление булевой функции f(x) = s · x mod 2 (скалярное произведение)
            f_x = np.dot(s, x) % 2

            # Вычисление нового значения для вспомогательного бита
            output_y = y ^ f_x  # y XOR f(x)

            # Индекс выхода (двоичное представление x, за которым следует новое y)
            output_index = int(''.join(map(str, x)) + str(output_y), 2)

            # Устанавливаем элемент матрицы (input_index, output_index) в 1
            oracle_matrix[input_index][output_index] = 1

    return oracle_matrix


if __name__ == '__main__':
    # Пример
    N = 3  # Длина скрытой двоичной строки s
    s = [1, 0, 0]  # Скрытый двоичный вектор, который мы хотим найти

    # Создаём квантовый симулятор с N + 1 кубитами (N для входа, 1 вспомогательный)
    sim = NQubitSimulator(N + 1)

    # Генерируем оракул на основе скрытой строки s
    oracle = generate_oracle_bernstein_vazirani(N, s)

    # Запускаем алгоритм Бернштейна-Вазирани
    result = bernstein_vazirani(sim, oracle)

    print(f'Скрытый двоичный вектор s: {s}')
    print(f'Измерено: {result}')

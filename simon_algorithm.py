import numpy as np
from symulator import NQubitSimulator
from constants import H
from sympy import Matrix


def simon_algorithm(simulator: NQubitSimulator, oracle) -> str:
    """
    Реализация алгоритма Саймона для поиска периода функции.
    simulator: NQubitSimulator — симулятор квантовой системы.
    oracle: np.ndarray — оракул функции с периодом.
    """
    simulator.reset()

    # Применяем Адамара только ко входным кубитам (первая половина кубитов)
    for i in range(simulator.dimension // 2):
        simulator.apply_single_qubit_gate(H, i)

    # Применяем оракул
    simulator.apply_n_qubit_gate(oracle)

    # Повторное применение Адамара ко входным кубитам
    for i in range(simulator.dimension // 2):
        simulator.apply_single_qubit_gate(H, i)

    # Измеряем все входные кубиты (только первую половину)
    result = []
    for i in range(simulator.dimension // 2):
        result.append(simulator.measure(i))

    return ''.join(map(str, result))


def generate_oracle_simon(N, s):
    """
    Генерация оракула для задачи Саймона.

    Аргументы:
    - N: число переменных (длина двоичной строки).
    - s: скрытая строка (битовый вектор), используемая для периодической функции f(x) = f(x ⊕ s).

    Возвращает:
    - Матрица 2^(2N) x 2^(2N), описывающая оракул.
    """
    matrix_size = 2 ** (2 * N)
    oracle_matrix = np.zeros((matrix_size, matrix_size), dtype=complex)

    # Проход по всем возможным значениям x
    for x in range(2 ** N):
        x_bin = f'{x:0{N}b}'  # бинарное представление x длиной N
        x_xor_s = x ^ int(''.join(map(str, s)), 2)  # x ⊕ s

        for y in range(2):
            # Индексы входов
            input_index = int(f'{x_bin}{y}', 2)
            output_index = int(f'{x_xor_s:0{N}b}{y}', 2)

            # Отображаем f(x) = f(x ⊕ s)
            oracle_matrix[input_index, output_index] = 1

    return oracle_matrix


def collect_results(simulator, oracle, num_runs=10):
    """Собирает несколько результатов измерений для алгоритма Саймона."""
    results = []
    for _ in range(num_runs):
        result = simon_algorithm(simulator, oracle)
        results.append(result)
    return results


def solve_simon_equations(results):
    """
    Решает систему линейных уравнений, чтобы найти скрытый период s на основе результатов измерений.
    Аргументы:
    - results: список результатов измерений (векторов).

    Возвращает:
    - Период s (в виде бинарной строки).
    """
    # Преобразуем результаты в бинарные векторы
    n = len(results[0])
    A = Matrix([list(map(int, result)) for result in results])

    # Решаем систему уравнений по модулю 2 (матрица A * s = 0)
    # Используем rref (reduced row echelon form) для приведения матрицы
    A_mod2 = A.applyfunc(lambda x: x % 2)
    rref_matrix, pivot_columns = A_mod2.rref()

    # Период — это ненулевое решение системы
    s = [0] * n
    for i, pivot in enumerate(pivot_columns):
        s[pivot] = 1

    return ''.join(map(str, s))


if __name__ == '__main__':
    # Пример использования алгоритма Саймона для поиска периода
    N = 2  # Число входных кубитов (длина строки x)
    s1 = [1, 0]  # Период s = "10"
    s2 = [1, 1]  # Период s = "11"

    # Генерация оракула для заданных периодов
    oracle_s1 = generate_oracle_simon(N, s1)
    oracle_s2 = generate_oracle_simon(N, s2)

    # Симуляция для системы из 2N кубитов (2 входных + 2 вспомогательных)
    nQubSim = NQubitSimulator(4)

    # Собираем результаты для ORACLE_S1
    print('ORACLE_S1 (s = 10):')
    results_s1 = collect_results(nQubSim, oracle_s1, num_runs=2)
    print(f'Results: {results_s1}')

    # Решаем уравнения для нахождения периода s
    period_s1 = solve_simon_equations(results_s1)
    print(f'Найденный период для ORACLE_S1: {period_s1}\n')

    # Собираем результаты для ORACLE_S2
    print('ORACLE_S2 (s = 11):')
    results_s2 = collect_results(nQubSim, oracle_s2, num_runs=2)
    print(f'Results: {results_s2}')

    # Решаем уравнения для нахождения периода s
    period_s2 = solve_simon_equations(results_s2)
    print(f'Найденный период для ORACLE_S2: {period_s2}')

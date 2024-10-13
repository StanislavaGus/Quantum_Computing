import numpy as np

from symulator import NQubitSimulator
from constants import H, X, HN
from deutsch_algorithm import ORACLE1, ORACLE2, ORACLE3, ORACLE4


# Алгоритм Дойча-Джозы
def deutsch_jozsa_algorithm(simulator: NQubitSimulator, oracle) -> bool:
    simulator.reset()

    simulator.apply_single_qubit_gate(X, (simulator.dimension - 1))  # устанавливаем последний кубит |1)

    # Применяем Адамара ко всем кубитам
    #simulator.apply_n_qubit_gate(HN(simulator.dimension))
    for i in range(simulator.dimension):
        simulator.apply_single_qubit_gate(H, i)

    # Применяем оракул
    simulator.apply_n_qubit_gate(oracle)

    # Применяем Адамара снова ко всем входным кубитам (кроме последнего)
    #simulator.apply_n_qubit_gate(HN(simulator.dimension))
    for i in range(simulator.dimension - 1):
        simulator.apply_single_qubit_gate(H, i)

    # Измеряем результат
    for i in range(simulator.dimension - 1):  # Измеряем только входные кубиты
        if simulator.measure(i) == 1:
            return "Balanced"
    return "Constant"



# Пример функций для трех входных параметров
#def bool_and_3(x) -> bool:
 #   """Функция "AND" для трех входных кубитов."""
  #  return x[0] and x[1] and x[2]

def bool_zero_3(x) -> bool:
    """Функция "0" для трех входных кубитов."""
    return False

def bool_or_3(x) -> bool:
    """Функция "OR" для трех входных кубитов."""
    return x[0] or x[1] or x[2]

def bool_xor_3(x) -> bool:
    """Функция "XOR" для трех входных кубитов."""
    return (x[0] != x[1]) != x[2]  # XOR между тремя входными кубитами


def generate_oracle_deutsch_jozsa(n: int, f) -> np.ndarray:
    """Генерация оракула для алгоритма Дойча-Джозы."""
    size = 2 ** (n + 1)  # Размер матрицы для n входных кубитов + 1 дополнительный
    oracle = np.eye(size, dtype=complex)  # Начинаем с тождественной матрицы

    for x in range(2 ** n):
        inputs = [(x >> i) & 1 for i in range(n)]  # Побитовый разбор входного числа
        output = f(inputs)  # Вызываем функцию f для получения результата
        if output:  # Если результат true, меняем соответствующий элемент оракула
            oracle[x, x] = -1  # Применяем фазовый сдвиг

    return oracle


if __name__ == '__main__':

    nQubSim = NQubitSimulator(2)

    print('ORACLE1: f(x) = 0')
    print(f'Result: {deutsch_jozsa_algorithm(nQubSim, ORACLE1)}')

    print('ORACLE2: f(x) = 1')
    print(f'Result: {deutsch_jozsa_algorithm(nQubSim, ORACLE2)}')

    print('ORACLE3: f(x) = x')
    print(f'Result: {deutsch_jozsa_algorithm(nQubSim, ORACLE3)}')

    print('ORACLE4: f(x) = !x')
    print(f'Result: {deutsch_jozsa_algorithm(nQubSim, ORACLE4)}')

    print("\n\n\n")

    nQubSim = NQubitSimulator(4)  # Симулятор с 4 кубитами, последний - дополнительный

    # Генерируем оракулы для каждой из функций
    oracle_zero = generate_oracle_deutsch_jozsa(3, bool_zero_3)
    oracle_or = generate_oracle_deutsch_jozsa(3, bool_or_3)
    oracle_xor = generate_oracle_deutsch_jozsa(3, bool_xor_3)


    # Запускаем алгоритм Дойча-Джозы для каждого оракула
    print("ORACLE (Zero for 3 inputs):")
    result = deutsch_jozsa_algorithm(nQubSim, oracle_zero)
    print(f"Result: {result}")

    print("ORACLE (OR for 3 inputs):")
    result = deutsch_jozsa_algorithm(nQubSim, oracle_or)
    print(f"Result: {result}")

    print("ORACLE (XOR for 3 inputs):")
    result = deutsch_jozsa_algorithm(nQubSim, oracle_xor)
    print(f"Result: {result}")



import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFT
from numpy.random import randint
from math import gcd
from fractions import Fraction


def check_if_power(N):
    """Проверяет, является ли число N целой степенью некоторого a^b.

    Алгоритм ищет значения `a` и `b`, такие что N = a^b.
    Если такие значения находятся, возвращает (True, a).
    Если N не является целой степенью, возвращает (False, None).
    """
    for b in range(2, int(np.log2(N)) + 1):
        a = round(N ** (1 / b))
        if a ** b == N:
            return True, a
    return False, None


def classical_modular_exponentiation(a, power, N):
    """Вычисляет (a^power) % N классическим методом.

    Используется для проверки делителей числа N на этапе
    классической обработки в алгоритме Шора.
    """
    return pow(a, power, N)


def add_inverse_qft(circuit, n):
    """Добавляет обратное квантовое преобразование Фурье (QFT) в квантовую схему.

    Обратное QFT переводит квантовое состояние обратно из фазы,
    чтобы измерение дало результат, который может быть интерпретирован классически.
    """
    circuit.append(QFT(num_qubits=n, inverse=True, do_swaps=True), range(n))


def create_period_finding_circuit(a, N, num_qubits):
    """Создает квантовую схему для поиска периода функции f(x) = a^x % N.

    Эта схема использует технику фазовой оценки для нахождения периода функции.
    1. Применяет операторы Адамара для перевода кубитов в суперпозицию.
    2. Использует специальную схему для модулярного возведения в степень.
    3. Добавляет обратное квантовое преобразование Фурье и измеряет результат.
    """
    # Создаем схему с кубитами для фазы и одного дополнительного кубита для хранения результата
    circuit = QuantumCircuit(2 * num_qubits, num_qubits)

    # Переводим кубиты первого регистра в суперпозицию
    circuit.h(range(num_qubits))

    # Добавляем оператор модульного возведения в степень, который действует как "черный ящик"
    modular_exp_circuit = create_modular_exponentiation_gate(a, N, num_qubits)
    circuit.append(modular_exp_circuit, range(num_qubits + 1))

    # Применяем обратное QFT для извлечения фазы, связанной с периодом функции
    add_inverse_qft(circuit, num_qubits)

    # Измеряем первый регистр
    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit


def create_modular_exponentiation_gate(a, N, num_qubits):
    """Создает квантовый оператор для модулярного возведения в степень.

    Эта функция представляет модулярное возведение в степень (a^(2^i) % N)
    как квантовый оператор, который управляет фазой основного кубита.
    """
    qc = QuantumCircuit(num_qubits + 1)
    for i in range(num_qubits):
        # Возведение числа `a` в степень 2^i, затем взятие остатка по модулю N
        power = (a ** (2 ** i)) % N
        if power > 0:
            # Если результат не нулевой, добавляем условные операции
            qc.cx(i, num_qubits)
            qc.rz(2 * np.pi * power / N, num_qubits)
            qc.cx(i, num_qubits)
    return qc.to_gate(label="ModularExp")


def extract_period_from_measurement(measurement_result, num_qubits):
    """Определяет период из измеренного результата квантовой схемы.

    Извлекает фазу измерения и определяет её период, используя дробное представление.
    Фаза интерпретируется как отношение, из которого можно найти период.
    """
    # Преобразуем результат из двоичного формата в десятичный
    decimal_result = int(measurement_result, 2)
    # Вычисляем фазу как дробь, используя битовую длину результата
    phase = decimal_result / (2 ** num_qubits)
    # Находим ближайшее рациональное представление фазы
    frac = Fraction(phase).limit_denominator()
    return frac.denominator  # Возвращаем знаменатель, который является искомым периодом


def shor_find_factors(N):
    """Основная функция алгоритма Шора для нахождения нетривиальных делителей числа N.

    1. Проверяет, является ли N четным или целой степенью.
    2. Если нет, начинает квантовый алгоритм поиска периода.
    3. Определяет возможные делители на основе найденного периода.
    """
    if N % 2 == 0:
        return [2]  # Если N четное, возвращаем делитель 2

    # Проверка, является ли N целой степенью
    is_power, base = check_if_power(N)
    if is_power:
        return [base]

    # Основная часть алгоритма Шора
    for attempt in range(20):  # Пробуем до 20 раз
        print(f"\nПопытка №{attempt + 1} для числа {N}")

        # Генерируем случайное число a, взаимно простое с N
        a = randint(2, N)
        if gcd(a, N) > 1:
            print(f"Найден делитель без квантового анализа: {gcd(a, N)}")
            return [gcd(a, N)]

        # Вычисляем необходимое количество кубитов
        num_qubits = int(np.ceil(np.log2(N)))

        # Создаем квантовую схему для нахождения периода
        circuit = create_period_finding_circuit(a, N, num_qubits)

        # Симулируем схему с увеличенным числом измерений для улучшения результатов
        simulator = Aer.get_backend('aer_simulator')
        transpiled_circuit = transpile(circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=4096)
        result = job.result()
        counts = result.get_counts()

        # Обрабатываем результаты измерений, начиная с наиболее частого
        measured_results = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        for measurement_result, freq in measured_results[:5]:  # Анализируем топ 5 вероятных измерений
            print(f"Измеренный результат: {measurement_result} (частота: {freq})")
            r = extract_period_from_measurement(measurement_result, num_qubits)
            print(f"Предполагаемый период r: {r}")

            # Проверяем, что период r четный и подходящий
            if r % 2 == 0 and classical_modular_exponentiation(a, r // 2, N) != -1 % N:
                # Вычисляем потенциальные делители
                factor1 = gcd(classical_modular_exponentiation(a, r // 2, N) - 1, N)
                factor2 = gcd(classical_modular_exponentiation(a, r // 2, N) + 1, N)

                # Проверяем, что оба фактора являются делителями числа N
                if factor1 > 1 and factor2 > 1 and factor1 * factor2 == N:
                    print(f"Найдены делители для числа {N}: {factor1} и {factor2}")
                    return list(set([factor1, factor2]))

    print("Не удалось найти нетривиальные делители.")
    return None


def execute_shor_on_numbers(numbers):
    """Запускает алгоритм Шора для каждого числа из переданного списка чисел."""
    results = {}
    for N in numbers:
        factors = shor_find_factors(N)
        results[N] = factors
        if factors:
            print(f"\nНетривиальные делители числа {N}: {factors}")
        else:
            print(f"\nНе удалось найти нетривиальные делители для числа {N}")
    return results


# Пример использования для массива чисел
test_numbers = [15, 77, 209, 221]
experiment_results = execute_shor_on_numbers(test_numbers)
